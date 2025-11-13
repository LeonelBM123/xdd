# -*- coding: utf-8 -*-
"""
API de Predicción de Ventas
Servicio REST para predecir probabilidades de venta de productos
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse  # (ni lo usas, pero lo dejo por si luego)
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import psycopg2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import datetime
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Variables globales para el modelo
modelo_entrenado = None
features_modelo = None
df_historico = None
ultima_actualizacion = None
accuracy_global = None


# -----------------------------------------------------------------
# MODELOS PYDANTIC (Respuestas)
# -----------------------------------------------------------------
class PrediccionProducto(BaseModel):
    producto_id: int
    probabilidad_venta: float
    se_vendera: bool


class PrediccionDia(BaseModel):
    fecha: str
    dia_semana: str
    total_productos_predichos: int
    productos: List[PrediccionProducto]


class RespuestaPrediccion(BaseModel):
    status: str
    mensaje: str
    fecha_inicio: Optional[str]
    fecha_fin: Optional[str]
    dias_predichos: int
    predicciones: List[PrediccionDia]


class EstadoModelo(BaseModel):
    status: str
    modelo_entrenado: bool
    ultima_actualizacion: Optional[str]
    total_registros_historicos: Optional[int]
    accuracy_modelo: Optional[float]


# -----------------------------------------------------------------
# FUNCIONES DE BASE DE DATOS
# -----------------------------------------------------------------
def get_db_connection():
    """Establece conexión con la base de datos PostgreSQL."""
    try:
        conn = psycopg2.connect(
            dbname="neondb",
            user="neondb_owner",
            password="npg_FBdE93ifxmvG",
            host="ep-wispy-cherry-a4ap6tji-pooler.us-east-1.aws.neon.tech",
            port="5432"
        )
        logger.info("Conexión a PostgreSQL exitosa")
        return conn
    except psycopg2.OperationalError as e:
        logger.error(f"Error al conectar a la base de datos: {e}")
        raise HTTPException(status_code=500, detail=f"Error de conexión a BD: {str(e)}")


def cargar_y_preparar_datos_diarios(conn):
    """Crea un DataFrame completo con registros diarios por producto."""
    logger.info("Cargando datos de ventas diarias...")

    sql_ventas_diarias = """
    SELECT 
        d.producto_id,
        DATE(v.fecha_venta) AS fecha,
        SUM(d.cantidad) AS total_vendido
    FROM 
        ventas_detalle_venta AS d
    JOIN 
        ventas_venta AS v ON d.venta_id = v.id
    GROUP BY 
        d.producto_id, DATE(v.fecha_venta);
    """
    df_ventas = pd.read_sql(sql_ventas_diarias, conn)
    df_ventas['fecha'] = pd.to_datetime(df_ventas['fecha'])

    sql_productos = "SELECT id FROM comercial_producto;"
    df_productos = pd.read_sql(sql_productos, conn)

    if df_ventas.empty or df_productos.empty:
        raise HTTPException(status_code=500, detail="No se encontraron datos de ventas o productos")

    logger.info("Creando scaffold de datos...")
    fecha_inicio = df_ventas['fecha'].min()
    fecha_fin = df_ventas['fecha'].max()

    date_range = pd.date_range(fecha_inicio, fecha_fin, freq='D')

    df_scaffold = pd.MultiIndex.from_product(
        [date_range, df_productos['id']],
        names=['fecha', 'producto_id']
    ).to_frame(index=False)

    df_completo = pd.merge(df_scaffold, df_ventas, on=['fecha', 'producto_id'], how='left')
    df_completo['total_vendido'] = df_completo['total_vendido'].fillna(0)
    df_completo['se_vendio'] = (df_completo['total_vendido'] > 0).astype(int)

    logger.info(f"Scaffold creado. Total de registros: {len(df_completo)}")
    return df_completo


# -----------------------------------------------------------------
# INGENIERÍA DE FEATURES
# -----------------------------------------------------------------
def ingenieria_de_features_diarias(df):
    """Crea features basados en el tiempo."""
    logger.info("Creando features...")

    df = df.sort_values(by=['producto_id', 'fecha'])

    df['ano'] = df['fecha'].dt.year
    df['mes'] = df['fecha'].dt.month
    df['dia_del_ano'] = df['fecha'].dt.dayofyear
    df['dia_de_la_semana'] = df['fecha'].dt.dayofweek
    df['es_fin_de_semana'] = (df['dia_de_la_semana'] >= 5).astype(int)

    g = df.groupby('producto_id')
    df['se_vendio_ayer'] = g['se_vendio'].shift(1)
    df['se_vendio_ultimos_7_dias'] = g['se_vendio'].shift(1).rolling(7, min_periods=1).sum()
    df['se_vendio_ultimos_30_dias'] = g['se_vendio'].shift(1).rolling(30, min_periods=1).sum()

    df = df.fillna(0)
    return df


# -----------------------------------------------------------------
# ENTRENAMIENTO DEL MODELO
# -----------------------------------------------------------------
def entrenar_modelo_clasificacion(df):
    """Entrena un RandomForestClassifier."""
    logger.info("Entrenando modelo...")

    df_train = df[df['ano'] < 2024]
    df_test = df[df['ano'] == 2024]

    if df_test.empty or df_train.empty:
        raise HTTPException(status_code=500, detail="No hay suficientes datos para entrenamiento")

    features = [
        'producto_id', 'ano', 'mes', 'dia_del_ano', 'dia_de_la_semana',
        'es_fin_de_semana', 'se_vendio_ayer', 'se_vendio_ultimos_7_dias',
        'se_vendio_ultimos_30_dias'
    ]

    target = 'se_vendio'

    X_train = df_train[features]
    y_train = df_train[target]
    X_test = df_test[features]
    y_test = df_test[target]

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    logger.info(f"Modelo entrenado. Accuracy: {accuracy:.3f}")

    return model, features, accuracy


# -----------------------------------------------------------------
# PREDICCIÓN
# -----------------------------------------------------------------
def predecir_siguientes_N_dias(model, features, df_historico_completo, n_dias=10):
    """Predice los N días futuros de forma iterativa."""
    logger.info(f"Prediciendo {n_dias} días futuros...")

    fecha_hoy = df_historico_completo['fecha'].max()
    datos_recientes = df_historico_completo[
        df_historico_completo['fecha'] > (fecha_hoy - datetime.timedelta(days=31))
    ].copy()

    todos_productos = df_historico_completo['producto_id'].unique()
    predicciones_futuras = []

    for i in range(1, n_dias + 1):
        fecha_a_predecir = fecha_hoy + datetime.timedelta(days=i)

        df_dia_futuro = pd.DataFrame({
            'fecha': [fecha_a_predecir] * len(todos_productos),
            'producto_id': todos_productos
        })

        df_dia_futuro['ano'] = df_dia_futuro['fecha'].dt.year
        df_dia_futuro['mes'] = df_dia_futuro['fecha'].dt.month
        df_dia_futuro['dia_del_ano'] = df_dia_futuro['fecha'].dt.dayofyear
        df_dia_futuro['dia_de_la_semana'] = df_dia_futuro['fecha'].dt.dayofweek
        df_dia_futuro['es_fin_de_semana'] = (df_dia_futuro['dia_de_la_semana'] >= 5).astype(int)

        temp_df = pd.concat([datos_recientes, df_dia_futuro], sort=False).sort_values(
            by=['producto_id', 'fecha']
        )

        g = temp_df.groupby('producto_id')
        temp_df['se_vendio_ayer'] = g['se_vendio'].shift(1)
        temp_df['se_vendio_ultimos_7_dias'] = g['se_vendio'].shift(1).rolling(7, min_periods=1).sum()
        temp_df['se_vendio_ultimos_30_dias'] = g['se_vendio'].shift(1).rolling(30, min_periods=1).sum()

        df_para_predecir = temp_df[temp_df['fecha'] == fecha_a_predecir].copy()
        X_futuro = df_para_predecir[features].fillna(0)

        probabilidades = model.predict_proba(X_futuro)[:, 1]

        df_dia_futuro['probabilidad_venta'] = probabilidades
        df_dia_futuro['se_vendio'] = (probabilidades > 0.5).astype(int)

        predicciones_futuras.append(df_dia_futuro)
        datos_recientes = pd.concat([datos_recientes, df_dia_futuro], sort=False)

    return pd.concat(predicciones_futuras, sort=False)


# -----------------------------------------------------------------
# LIFESPAN (reemplazo de @app.on_event("startup"))
# -----------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Se ejecuta al iniciar la app (startup) y al cerrarse (shutdown),
    sin usar el decorador @app.on_event.
    """
    global modelo_entrenado, features_modelo, df_historico, ultima_actualizacion, accuracy_global

    try:
        logger.info("Inicializando modelo (lifespan)...")
        conn = get_db_connection()
        df_base = cargar_y_preparar_datos_diarios(conn)
        df_historico = ingenieria_de_features_diarias(df_base)
        modelo_entrenado, features_modelo, accuracy_global = entrenar_modelo_clasificacion(df_historico)
        ultima_actualizacion = datetime.datetime.now().isoformat()
        conn.close()
        logger.info("Modelo inicializado correctamente (lifespan)")
    except Exception as e:
        logger.error(f"Error al inicializar modelo en lifespan: {e}")

    # Aquí la app ya está levantada
    yield

    # Aquí podrías cerrar recursos si tuvieras (BD persistente, etc.)
    logger.info("Apagando API de Predicción de Ventas")


# -----------------------------------------------------------------
# INICIALIZAR FastAPI CON LIFESPAN
# -----------------------------------------------------------------
app = FastAPI(
    title="API de Predicción de Ventas",
    description="Servicio para predecir probabilidades de venta de productos",
    version="1.0.0",
    lifespan=lifespan,
)


# -----------------------------------------------------------------
# ENDPOINTS DE LA API
# -----------------------------------------------------------------
@app.get("/", tags=["General"])
async def root():
    """Endpoint raíz con información de la API."""
    return {
        "mensaje": "API de Predicción de Ventas",
        "version": "1.0.0",
        "endpoints": {
            "predecir": "/api/predecir",
            "estado": "/api/estado",
            "reentrenar": "/api/reentrenar"
        }
    }


@app.get("/api/estado", response_model=EstadoModelo, tags=["Modelo"])
async def obtener_estado():
    """Obtiene el estado actual del modelo."""
    return EstadoModelo(
        status="ok",
        modelo_entrenado=modelo_entrenado is not None,
        ultima_actualizacion=ultima_actualizacion,
        total_registros_historicos=len(df_historico) if df_historico is not None else None,
        accuracy_modelo=float(accuracy_global) if accuracy_global is not None else None,
    )


@app.post("/api/reentrenar", tags=["Modelo"])
async def reentrenar_modelo():
    """Reentrena el modelo con los datos más recientes."""
    global modelo_entrenado, features_modelo, df_historico, ultima_actualizacion, accuracy_global

    try:
        conn = get_db_connection()
        df_base = cargar_y_preparar_datos_diarios(conn)
        df_historico = ingenieria_de_features_diarias(df_base)
        modelo_entrenado, features_modelo, accuracy_global = entrenar_modelo_clasificacion(df_historico)
        ultima_actualizacion = datetime.datetime.now().isoformat()
        conn.close()

        return {
            "status": "ok",
            "mensaje": "Modelo reentrenado correctamente",
            "accuracy": float(accuracy_global),
            "fecha_actualizacion": ultima_actualizacion
        }
    except Exception as e:
        logger.error(f"Error al reentrenar: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/predecir", response_model=RespuestaPrediccion, tags=["Predicciones"])
async def predecir_ventas(
    dias: int = Query(default=10, ge=1, le=30, description="Número de días a predecir (1-30)"),
    top_productos: int = Query(default=5, ge=1, le=50, description="Top N productos por día"),
    umbral_probabilidad: float = Query(default=0.0, ge=0.0, le=1.0, description="Probabilidad mínima para incluir producto")
):
    """
    Predice las ventas para los próximos N días.

    - **dias**: Número de días a predecir (1-30)
    - **top_productos**: Cantidad de productos a mostrar por día
    - **umbral_probabilidad**: Filtrar productos con probabilidad mayor a este valor
    """
    if modelo_entrenado is None:
        raise HTTPException(status_code=503, detail="Modelo no está entrenado. Intente más tarde.")

    try:
        predicciones_df = predecir_siguientes_N_dias(
            modelo_entrenado,
            features_modelo,
            df_historico,
            n_dias=dias
        )

        predicciones_por_dia = []
        dias_semana = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']

        for fecha in sorted(predicciones_df['fecha'].unique()):
            df_dia = predicciones_df[predicciones_df['fecha'] == fecha]
            df_dia = df_dia[df_dia['probabilidad_venta'] >= umbral_probabilidad]
            df_dia = df_dia.nlargest(top_productos, 'probabilidad_venta')

            fecha_dt = pd.to_datetime(fecha)

            productos = [
                PrediccionProducto(
                    producto_id=int(row['producto_id']),
                    probabilidad_venta=round(float(row['probabilidad_venta']), 4),
                    se_vendera=bool(row['probabilidad_venta'] > 0.5)
                )
                for _, row in df_dia.iterrows()
            ]

            predicciones_por_dia.append(PrediccionDia(
                fecha=fecha_dt.strftime('%Y-%m-%d'),
                dia_semana=dias_semana[fecha_dt.dayofweek],
                total_productos_predichos=len(productos),
                productos=productos
            ))

        fecha_inicio = predicciones_por_dia[0].fecha if predicciones_por_dia else None
        fecha_fin = predicciones_por_dia[-1].fecha if predicciones_por_dia else None

        return RespuestaPrediccion(
            status="ok",
            mensaje=f"Predicción generada exitosamente para {dias} días",
            fecha_inicio=fecha_inicio,
            fecha_fin=fecha_fin,
            dias_predichos=len(predicciones_por_dia),
            predicciones=predicciones_por_dia
        )

    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        raise HTTPException(status_code=500, detail=f"Error al generar predicción: {str(e)}")


# -----------------------------------------------------------------
# EJECUTAR CON: uvicorn predicciones6:app --reload
# -----------------------------------------------------------------

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 01:04:19 2025

@author: PC
"""

import pandas as pd
import psycopg2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import datetime

# -----------------------------------------------------------------
# PASO 1: CONEXIÓN A LA BASE DE DATOS
# -----------------------------------------------------------------
def get_db_connection():
    """Establece conexión con la base de datos PostgreSQL."""
    try:
        conn = psycopg2.connect(
            dbname="dataset2",  # <-- RELLENA ESTO
            user="postgres",        # <-- RELLENA ESTO
            password="leoncio123",  # <-- RELLENA ESTO
            host="localhost",         # <-- RELLENA ESTO (usualmente localhost)
            port="5432"               # <-- RELLENA ESTO
        )
        print("¡Conexión a PostgreSQL exitosa!")
        return conn
    except psycopg2.OperationalError as e:
        print(f"Error al conectar a la base de datos: {e}")
        return None

# -----------------------------------------------------------------
# PASO 2: CARGAR Y PREPARAR DATOS DIARIOS (MUY IMPORTANTE)
# -----------------------------------------------------------------
def cargar_y_preparar_datos_diarios(conn):
    """
    Crea un DataFrame completo, con un registro por producto y por día, 
    incluyendo los días CON CERO ventas.
    """
    print("Cargando datos de ventas diarias...")
    
    # 1. Obtener las ventas reales agrupadas por día y producto
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
    
    # 2. Obtener todos los productos únicos
    sql_productos = "SELECT id FROM comercial_producto;"
    df_productos = pd.read_sql(sql_productos, conn)
    
    if df_ventas.empty or df_productos.empty:
        print("Error: No se encontraron datos de ventas o productos.")
        return None

    # 3. Crear el "Scaffold" (parrilla completa de días y productos)
    # Esto es CRUCIAL para que el modelo aprenda de los días "cero"
    print("Creando 'scaffold' de datos (días-producto)...")
    fecha_inicio = df_ventas['fecha'].min()
    # Usamos la fecha de hoy (en la simulación, sería la última fecha de 2024)
    fecha_fin = df_ventas['fecha'].max() 
    
    date_range = pd.date_range(fecha_inicio, fecha_fin, freq='D')
    
    df_scaffold = pd.MultiIndex.from_product(
        [date_range, df_productos['id']], 
        names=['fecha', 'producto_id']
    ).to_frame(index=False)

    # 4. Unir (Merge) las ventas reales al scaffold
    df_completo = pd.merge(
        df_scaffold, 
        df_ventas, 
        on=['fecha', 'producto_id'], 
        how='left'
    )
    
    # 5. Rellenar NaNs (días sin venta) con 0
    df_completo['total_vendido'] = df_completo['total_vendido'].fillna(0)
    
    # 6. Crear el objetivo (y): se_vendio (1 si > 0, 0 si no)
    df_completo['se_vendio'] = (df_completo['total_vendido'] > 0).astype(int)
    
    print(f"Scaffold creado. Total de registros (días * productos): {len(df_completo)}")
    
    return df_completo

# -----------------------------------------------------------------
# PASO 3: INGENIERÍA DE FEATURES DIARIAS
# -----------------------------------------------------------------
def ingenieria_de_features_diarias(df):
    """Crea features basados en el tiempo (lags, rolling)."""
    print("Iniciando Ingeniería de Features Diarias...")
    
    # Asegurar el orden correcto
    df = df.sort_values(by=['producto_id', 'fecha'])
    
    # Características de Tiempo
    df['ano'] = df['fecha'].dt.year
    df['mes'] = df['fecha'].dt.month
    df['dia_del_ano'] = df['fecha'].dt.dayofyear
    df['dia_de_la_semana'] = df['fecha'].dt.dayofweek # Lunes=0, Domingo=6
    df['es_fin_de_semana'] = (df['dia_de_la_semana'] >= 5).astype(int)
    
    # Características de Lag (Tendencia)
    # Agrupamos por producto para que el lag no "salte" a otro producto
    g = df.groupby('producto_id')
    
    # Lag 1: ¿Se vendió ayer? (La feature más importante)
    df['se_vendio_ayer'] = g['se_vendio'].shift(1)
    
    # Rolling 7: ¿Cuántas VECES se vendió en los últimos 7 días?
    # (shift(1) es para no incluir el día actual)
    df['se_vendio_ultimos_7_dias'] = g['se_vendio'].shift(1).rolling(7, min_periods=1).sum()
    
    # Rolling 30: ¿Cuántas VECES se vendió en los últimos 30 días?
    df['se_vendio_ultimos_30_dias'] = g['se_vendio'].shift(1).rolling(30, min_periods=1).sum()
    
    # Llenar los NaNs iniciales (ej. el primer día no tiene "ayer")
    df = df.fillna(0)
    
    print(df.head())
    return df

# -----------------------------------------------------------------
# PASO 4: ENTRENAR MODELO DE CLASIFICACIÓN
# -----------------------------------------------------------------
def entrenar_modelo_clasificacion(df):
    """Entrena un RandomForestClassifier."""
    print("\n--- Iniciando Entrenamiento del Clasificador ---")
    
    # Dividir por tiempo
    df_train = df[df['ano'] < 2024]
    df_test = df[df['ano'] == 2024]

    if df_test.empty or df_train.empty:
        print("Error: No hay suficientes datos para la división 2020-2023 (Train) y 2024 (Test).")
        return None, None

    # Definir features y target
    features = [
        'producto_id', 'ano', 'mes', 'dia_del_ano', 'dia_de_la_semana', 
        'es_fin_de_semana', 'se_vendio_ayer', 'se_vendio_ultimos_7_dias',
        'se_vendio_ultimos_30_dias'
    ]
    
    target = 'se_vendio' # Nuestro objetivo 1 o 0

    X_train = df_train[features]
    y_train = df_train[target]
    
    X_test = df_test[features]
    y_test = df_test[target]
    
    # Modelo: RandomForestClassifier
    # class_weight='balanced' es MUY importante porque (probablemente)
    # hay muchos más días con "0" (No Venta) que con "1" (Venta).
    model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        n_jobs=-1,
        class_weight='balanced'
    )
    
    print("Entrenando RandomForestClassifier...")
    model.fit(X_train, y_train)
    
    # Evaluación
    print("\n--- Evaluación en Datos de Prueba (Año 2024) ---")
    y_pred = model.predict(X_test)
    
    print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.3f}")
    print(classification_report(y_test, y_pred))
    
    return model, features

# -----------------------------------------------------------------
# PASO 5: PREDECIR LOS SIGUIENTES N DÍAS (ITERATIVO)
# -----------------------------------------------------------------
def predecir_siguientes_N_dias(model, features, df_historico_completo, n_dias=10):
    """
    Predice los N días futuros de forma iterativa, 
    alimentando la predicción de un día como feature para el siguiente.
    """
    print(f"\n--- Iniciando Predicción Iterativa para {n_dias} Días Futuros ---")
    
    # 1. Preparar el DataFrame de entrada (los últimos 30 días de historia)
    # (Necesitamos 30 días para calcular 'se_vendio_ultimos_30_dias')
    fecha_hoy = df_historico_completo['fecha'].max()
    datos_recientes = df_historico_completo[
        df_historico_completo['fecha'] > (fecha_hoy - datetime.timedelta(days=31))
    ].copy()
    
    todos_productos = df_historico_completo['producto_id'].unique()
    predicciones_futuras = []

    for i in range(1, n_dias + 1):
        fecha_a_predecir = fecha_hoy + datetime.timedelta(days=i)
        
        # 2. Crear el 'scaffold' para el día que queremos predecir
        df_dia_futuro = pd.DataFrame({
            'fecha': [fecha_a_predecir] * len(todos_productos),
            'producto_id': todos_productos
        })
        
        # 3. Calcular las features para este día futuro
        
        # Features de tiempo (fáciles)
        df_dia_futuro['ano'] = df_dia_futuro['fecha'].dt.year
        df_dia_futuro['mes'] = df_dia_futuro['fecha'].dt.month
        df_dia_futuro['dia_del_ano'] = df_dia_futuro['fecha'].dt.dayofyear
        df_dia_futuro['dia_de_la_semana'] = df_dia_futuro['fecha'].dt.dayofweek
        df_dia_futuro['es_fin_de_semana'] = (df_dia_futuro['dia_de_la_semana'] >= 5).astype(int)
        
        # Features de Lag (requieren los datos 'recientes')
        # Unimos los datos recientes para calcular los lags
        # Nota: Esto es complejo, lo simplificaremos uniendo y calculando
        
        temp_df = pd.concat([datos_recientes, df_dia_futuro], sort=False).sort_values(by=['producto_id', 'fecha'])
        
        # Recalculamos los lags (igual que en la función de ingeniería)
        g = temp_df.groupby('producto_id')
        temp_df['se_vendio_ayer'] = g['se_vendio'].shift(1)
        temp_df['se_vendio_ultimos_7_dias'] = g['se_vendio'].shift(1).rolling(7, min_periods=1).sum()
        temp_df['se_vendio_ultimos_30_dias'] = g['se_vendio'].shift(1).rolling(30, min_periods=1).sum()
        
        # 4. Filtrar solo las filas del día que queremos predecir (ya tienen los lags)
        df_para_predecir = temp_df[temp_df['fecha'] == fecha_a_predecir].copy()
        
        # 5. Predecir
        X_futuro = df_para_predecir[features].fillna(0) # Rellenar NaNs por si acaso
        
        # Predecir PROBABILIDAD (predict_proba)
        # Nos da [prob_de_0, prob_de_1]
        probabilidades = model.predict_proba(X_futuro)[:, 1]
        
        df_dia_futuro['probabilidad_venta'] = probabilidades
        # Creamos una columna 'se_vendio' (predicha) para el siguiente bucle
        # (Asumimos que se vende si la prob > 50%)
        df_dia_futuro['se_vendio'] = (probabilidades > 0.5).astype(int)
        
        # 6. Guardar resultados y actualizar datos_recientes para el siguiente bucle
        predicciones_futuras.append(df_dia_futuro)
        datos_recientes = pd.concat([datos_recientes, df_dia_futuro], sort=False)

    return pd.concat(predicciones_futuras, sort=False)

# -----------------------------------------------------------------
# EJECUCIÓN PRINCIPAL
# -----------------------------------------------------------------
if __name__ == "__main__":
    conn = get_db_connection()
    
    if conn:
        df_base = cargar_y_preparar_datos_diarios(conn)
        
        if df_base is not None:
            df_features = ingenieria_de_features_diarias(df_base)
            
            modelo_clasificador, lista_features = entrenar_modelo_clasificacion(df_features)
            
            if modelo_clasificador:
                predicciones = predecir_siguientes_N_dias(
                    modelo_clasificador, 
                    lista_features, 
                    df_features, 
                    n_dias=10 # <- Predice 10 días
                )
                
                # --- Mostrar los resultados ---
                print("\n\n--- PREDICCIÓN DE VENTAS (PRÓXIMOS 10 DÍAS) ---")
                
                for fecha in predicciones['fecha'].unique():
                    fecha_str = pd.to_datetime(fecha).strftime('%Y-%m-%d (%A)')
                    print(f"\n--- {fecha_str} ---")
                    
                    # Top 5 productos con MÁS probabilidad de venderse
                    top_5 = predicciones[predicciones['fecha'] == fecha].nlargest(5, 'probabilidad_venta')
                    
                    if top_5.empty:
                        print(" (No se esperan ventas significativas)")
                    else:
                        for _, row in top_5.iterrows():
                            print(f"  - Producto ID: {int(row['producto_id']):<3} | Probabilidad: {row['probabilidad_venta']*100:5.1f}%")

        conn.close()
        print("\nConexión a la base de datos cerrada.")
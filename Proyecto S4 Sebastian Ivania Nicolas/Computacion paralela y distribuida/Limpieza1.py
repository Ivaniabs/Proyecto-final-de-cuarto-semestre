import multiprocessing as mp
import os
import time
import math
import pandas as pd
from pathlib import Path

Base_Dir = Path(__file__).resolve().parent.parent
Ruta_Dataset_Original = Base_Dir/"Datos"/"Dataset Original.csv"

#Columnas del Dataset
Columna_Radiacion_Solar = "RADIACIÓN SOLAR (w/m2)"
Columnas_Numericas_Restantes = ["DIRECCIÓN DEL VIENTO (m/s)", "VELOCIDAD DEL VIENTO (m/s)", "TEMPERATURA (°C)", "HUMEDAD (%)", "PRESIÓN (hPa)", "PRECIPITACIÓN (mm)"]

Columnas_Numericas_Completas = [Columna_Radiacion_Solar] + Columnas_Numericas_Restantes

def Convertir_Fechas(Fecha_Original):
    En_Texto = Fecha_Original.astype(str).str.strip()
    
    #Se uso el formato dia/mes/año hora:minutos
    Fechas = pd.to_datetime(En_Texto, format = "%d/%m/%Y %H:%M", errors = "coerce")
    
    #Se revisa con el segundo formato año-mes-dia hora:minuti:segundo
    Falta_por_Convertir = Fechas.isna()
    
    if Falta_por_Convertir.any():
        Fechas.loc[Falta_por_Convertir] = pd.to_datetime(
            En_Texto.loc[Falta_por_Convertir], format = "%Y-%m-%d %H:%M:%S", errors = "coerce"
        )
    
    #Segunda revision con formato dia/mes/año hora:minuto:segundos
    Falta_por_Convertir_2 = Fechas.isna()
    
    if Falta_por_Convertir_2.any():
        Fechas.loc[Falta_por_Convertir_2] = pd.to_datetime(
        En_Texto.loc[Falta_por_Convertir_2], format = "%d/%m/%Y %H:%M:%S", errors = "coerce"
    )
    
    return Fechas

def Limpieza_Bloque(Bloque):
    
    Datos = Bloque.copy()
    
    #Quitar espacios
    Datos.columns = [str(columna).strip() for columna in Datos.columns]
    
    #Eliminar columna de radiacion solar
    if "RADIACIÓN SOLAR (w/m2)" in Datos.columns:
            Datos = Datos.drop(columns=["RADIACIÓN SOLAR (w/m2)"])
    
    #Modificar columnas de texto
    Datos["ESTACION"] = Datos["ESTACION"].astype(str).str.strip().str.upper()
    Datos["FECHA CONVERTIDA"] = Convertir_Fechas(Datos["FECHA"])
    
    for columna in Columnas_Numericas_Completas:
        if columna in Datos.columns:
            Datos[columna] = pd.to_numeric(
                Datos[columna].astype(str).str.strip(), errors = "coerce"
            )
            
    Columnas_sin_Nulos = ["FECHA CONVERTIDA", "ESTACION"] + [c for c in Columnas_Numericas_Restantes if c in Datos.columns]
    
    Estacion_Valida = (
        Datos["ESTACION"].notna() & (Datos["ESTACION"] != "") & (Datos["ESTACION"] != "NAN")
    )
    Datos = Datos[Estacion_Valida].dropna(subset=Columnas_sin_Nulos).copy()
    
    #Separar fecha y horas
    Datos["AÑO"] = Datos["FECHA CONVERTIDA"].dt.year
    Datos["MES"] = Datos["FECHA CONVERTIDA"].dt.month
    Datos["HORA"] = Datos["FECHA CONVERTIDA"].dt.hour
    
    #Eliminar duplicados exactos
    Datos = Datos.drop_duplicates()
    
    return Datos

if __name__ == "__main__":
    print("1. Leer el archivo original", flush = True)
    Datos_Originales = pd.read_csv(Ruta_Dataset_Original, sep = ";", encoding = "utf-8-sig", dtype = str)
    
    print(f"Filas cargadas: {len(Datos_Originales)}", flush = True)
    
    Repeticiones = 5
    
    print("2. Proceso secuencial", flush = True)
    Tiempos_Secuenciales = []
    
    for repeticion in range(Repeticiones):
        Inicio_Secuencial = time.perf_counter()
        Resultado_Secuencial = Limpieza_Bloque(Datos_Originales)
        Tiempo_Secuencial = time.perf_counter() - Inicio_Secuencial
        Tiempos_Secuenciales.append(Tiempo_Secuencial)
    
    Tiempo_Secuencial = sum(Tiempos_Secuenciales)/Repeticiones
    print(f"Tiempo del proceso secuencial: {Tiempo_Secuencial: .4f} segundos", flush = True)
    
    #Procesis
    Cantidad_de_Procesos = min(4, os.cpu_count() or 2)
    Total_Filas = len(Datos_Originales)
    Filas_por_Bloque = math.ceil(Total_Filas/Cantidad_de_Procesos)
    
    Bloques = [
        Datos_Originales.iloc[i : i + Filas_por_Bloque].copy()
        for i in range(0, Total_Filas, Filas_por_Bloque)
    ]
    
    print(f"3. Proceso Paralelo con {Cantidad_de_Procesos} al tiempo", flush = True)
    
    Tiempos_Paralelos = []
    
    for repeticion in range(Repeticiones):
        Inicio_Paralelo = time.perf_counter()
    
        with mp.Pool(processes = Cantidad_de_Procesos) as Grupo_Procesos:
            Bloques_Limpios = Grupo_Procesos.map(Limpieza_Bloque, Bloques)
    
        Resultado_Paralelo = pd.concat(Bloques_Limpios, ignore_index = True)
        Resultado_Paralelo = Resultado_Paralelo.drop_duplicates()
        Tiempo_Paralelo = time.perf_counter() - Inicio_Paralelo
        Tiempos_Paralelos.append(Tiempo_Paralelo)
    
    Tiempo_Paralelo = sum(Tiempos_Paralelos)/Repeticiones
    
    Aceleracion = Tiempo_Secuencial/Tiempo_Paralelo if Tiempo_Paralelo > 0 else 0.0
    Eficiencia = Aceleracion/Cantidad_de_Procesos if Cantidad_de_Procesos > 0 else 0.0
    
    #Resultados
    print("\n 4. Resultados", flush = True)
    print(f"Filas limpias: {len(Resultado_Paralelo)}", flush = True)
    print(f"Tiempo promedio secuencial: {Tiempo_Secuencial: .4f} segundos", flush = True)
    print(f"Tiempo promedio paralelo: {Tiempo_Paralelo: .4f} segundos", flush = True)
    print(f"Aceleracion o Speedup: {Aceleracion: .2f} x", flush = True)
    print(f"Eficiencia: {Eficiencia: .2%}", flush = True)
    
    #Dataset limpio final
    Resultado_Paralelo.to_csv( 
        Base_Dir/"Datos"/"-Dataset Limpio2New.csv",
        index=False,
        encoding="utf-8-sig",
    )

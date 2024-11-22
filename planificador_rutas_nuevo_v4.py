import streamlit as st
import pandas as pd
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import folium
from streamlit_folium import st_folium
import numpy as np
from io import BytesIO
import re
import io
import concurrent.futures

# Título de la aplicación
st.title('Planificador de rutas 3.0')

# Cargar el archivo Excel
uploaded_file = st.file_uploader("Por favor, carga un archivo Excel con las direcciones de la ruta a planificar", type=["xlsx", "xls"])

@st.cache_data
def cargar_excel(uploaded_file):
    return pd.read_excel(uploaded_file)

if uploaded_file is not None:
    # Cargar datos del archivo Excel
    datos = cargar_excel(uploaded_file)
    st.write("Archivo Excel cargado correctamente!")

    # Convertir DataFrame a CSV
    csv_buffer = BytesIO()
    direcciones = datos.to_csv(csv_buffer, sep=';', encoding='latin1', index=False)
    csv_buffer.seek(0)

    # Cargar el archivo CSV convertido en memoria para continuar el procesamiento
    csv_buffer.seek(0)  # Reiniciar el buffer

# Inicializar el geocodificador Nominatim con RateLimiter para manejar mejor las solicitudes
def iniciar_geolocator():
    geolocator = Nominatim(user_agent="tu_aplicacion_de_geocodificacion")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1, max_retries=3)
    return geocode

geolocator = iniciar_geolocator()

# Función para limpiar la dirección
def limpiar_direccion(direccion):
    direccion_limpia = re.sub(r"(piso|dpto|departamento|apartamento|#|esc|izq|der|bajo|ptal|portal(.*\s?[^\w\s,]+))", "", direccion, flags=re.IGNORECASE)
    if not direccion_limpia.strip().startswith("C/"):
        direccion_limpia = "C/ " + direccion_limpia.strip()
    direccion_limpia = re.sub(r"[^a-zA-Z0-9ÁÉÍÓÚáéíóúñÑ\s,\/]", "", direccion_limpia)
    direccion_limpia = re.sub(r"\s+", " ", direccion_limpia)
    direccion_limpia = re.sub(r"\s*,\s*", ", ", direccion_limpia)
    patron = re.compile(r"C/\s*([A-Za-z0-9\s]+),\s*(\d+)[^,]*,\s*([A-Za-z\s]+)$", flags=re.IGNORECASE)
    match = patron.match(direccion_limpia)
    if match:
        calle = match.group(1).strip()
        numero = match.group(2).strip()
        ciudad = match.group(3).strip()
        return f"C/ {calle}, {numero}, {ciudad}"
    print(f"Dirección {direccion_limpia} no coincide con el formato esperado.")
    return None

# Definir la función
def sustituir_ñ(texto):
    texto = str(texto) if texto is not None else ''
    texto = re.sub(r'ñ', 'n', texto)
    texto = re.sub(r'Ñ', 'N', texto)
    texto = re.sub(r'SSR', 'SAN SEBASTIÁN DE LOS REYES, MADRID', texto)
    texto = re.sub(r'AVDA', 'AVENIDA', texto)
    texto = re.sub(r'CTRA\.\s*', 'CARRETERA ', texto)
    texto = re.sub(r',\s*', ', ', texto)
    texto = re.sub(r'CASTILLA LA MANCHA', 'DE CASTILLA-LA MANCHA', texto)
    return texto

# Función para obtener coordenadas
@st.cache_data
def obtener_coordenadas(direccion):
    try:
        location = geolocator(direccion, country_codes='es', timeout=20)
        if location:
            return (location.latitude, location.longitude)
        else:
            return (None, None)
    except Exception as e:
        print(f"Error obteniendo coordenadas para {direccion}: {e}")
        return (None, None)

# Función para obtener coordenadas en paralelo
def obtener_coordenadas_parallel(direcciones):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        resultados = list(executor.map(obtener_coordenadas, direcciones))
    return resultados

# Cálculo de la distancia con validación de None
def haversine(lat1, lon1, lat2, lon2):
    if None in [lat1, lon1, lat2, lon2]:
        return None
    R = 6371
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# Procesamiento, limpieza y carga de datos
if uploaded_file is not None:
    direcciones = cargar_excel(uploaded_file)
    st.write("Archivo cargado correctamente!")

    # Transformaciones columnas de DataFrame
    df_1 = pd.DataFrame(direcciones)
    df_1['NUMERO'] = df_1['NUMERO'].fillna(0).astype(int).astype(str)
    df_3 = df_1['DIRECCION COMPLETA'] + ',' + df_1['NUMERO'].fillna(0).astype(str) + ',' + df_1['CIUDAD']
    df_3 = pd.DataFrame(df_3, columns=['DIRECCION_COMPLETA'])
    df_4 = pd.concat([df_1, df_3], axis=1)
    df_4['AVISO'] = 'Sin Aviso'
    df_4 = df_4.drop('DIRECCION COMPLETA', axis=1)

    # Asegurarse de que todos los valores en la columna DIRECCION_COMPLETA sean cadenas
    df_4['DIRECCION_COMPLETA'] = df_4['DIRECCION_COMPLETA'].astype(str)

    # Aplicar la función de limpiar_direccion a la columna DIRECCION_COMPLETA
    df_4['DIRECCION_COMPLETA'] = df_4['DIRECCION_COMPLETA'].apply(sustituir_ñ)

    # Inicializar las columnas Latitud y Longitud
    df_4['Latitud'] = np.nan
    df_4['Longitud'] = np.nan

    # Campo de entrada de texto en lugar de selectbox para que el usuario introduzca la dirección manualmente
    direccion_seleccionada = st.text_input("Introduce la dirección de punto de partida: Calle, Número, Ciudad")

    if direccion_seleccionada:
        df_filtrado = df_4[df_4['DIRECCION_COMPLETA'] == direccion_seleccionada]
        if df_filtrado.empty:
            nueva_fila = pd.DataFrame({"DIRECCION_COMPLETA": [direccion_seleccionada]})
            df_4 = pd.concat([nueva_fila, df_4], ignore_index=True)
            st.write(f"Dirección '{direccion_seleccionada}' añadida a la lista de direcciones.")
        else:
            st.write(f"Datos filtrados para la dirección: {direccion_seleccionada}")

    if direccion_seleccionada not in df_4['DIRECCION_COMPLETA'].values:
        nueva_fila = pd.DataFrame({"DIRECCION_COMPLETA": [direccion_seleccionada]})
        df_4 = pd.concat([nueva_fila, df_4], ignore_index=True)

    df_4['Orden'] = df_4['DIRECCION_COMPLETA'].apply(lambda x: 0 if x == direccion_seleccionada else 1)
    df_4 = df_4.sort_values(by='Orden').drop(columns='Orden').reset_index(drop=True)

    # Diccionario para almacenar coordenadas ya obtenidas
    coordenadas_cache = {}

    # Función para obtener coordenadas con caché
    def obtener_coordenadas_con_cache(direccion):
        if direccion in coordenadas_cache:
            return coordenadas_cache[direccion]
        coordenadas = obtener_coordenadas(direccion)
        coordenadas_cache[direccion] = coordenadas
        return coordenadas

    # Aplicar la función para obtener latitud y longitud en paralelo solo para nuevas direcciones
    nuevas_direcciones = df_4[df_4['Latitud'].isna() | df_4['Longitud'].isna()]['DIRECCION_COMPLETA']
    if not nuevas_direcciones.empty:
        nuevas_coordenadas = obtener_coordenadas_parallel(nuevas_direcciones)
        df_4.loc[nuevas_direcciones.index, ['Latitud', 'Longitud']] = nuevas_coordenadas

    # Función y ordenamiento
    def ordenar_por_proximidad(df_ordenado):
        puntos_restantes = list(df_4.index)
        orden_recorrido = [puntos_restantes.pop(0)]

        while puntos_restantes:
            ultimo_punto = orden_recorrido[-1]
            lat_actual, lon_actual = df_4.loc[ultimo_punto, 'Latitud'], df_4.loc[ultimo_punto, 'Longitud']

            distancia_minima = float('inf')
            siguiente_punto = None

            for punto in puntos_restantes:
                lat_punto, lon_punto = df_4.loc[punto, 'Latitud'], df_4.loc[punto, 'Longitud']
                distancia = haversine(lat_actual, lon_actual, lat_punto, lon_punto)

                if distancia < distancia_minima:
                    distancia_minima = distancia
                    siguiente_punto = punto

            orden_recorrido.append(siguiente_punto)
            puntos_restantes.remove(siguiente_punto)

        df_ordenado = df_4.loc[orden_recorrido].reset_index(drop=True)
        df_ordenado['Distancia'] = 0.0
        for i in range(1, len(df_ordenado)):
            lat_anterior, lon_anterior = df_ordenado.loc[i-1, 'Latitud'], df_ordenado.loc[i-1, 'Longitud']
            lat_actual, lon_actual = df_ordenado.loc[i, 'Latitud'], df_ordenado.loc[i, 'Longitud']
            df_ordenado.loc[i, 'Distancia'] = haversine(lat_anterior, lon_anterior, lat_actual, lon_actual)

        return df_ordenado

    # Aplicar la función de ordenación al DataFrame después de obtener las coordenadas
    df_5_ordenado = ordenar_por_proximidad(df_4)

    # Asignar avisos cada 9 OT/CASO
    def asignar_avisos(df):
        pendientes = df
        df_5_ordenado['AVISO'] = np.nan
        aviso_num = 1
        for i in range(0, len(pendientes), 9):
            df_5_ordenado.loc[pendientes.index[i:i+9], 'AVISO'] = aviso_num
            aviso_num += 1
        return df_5_ordenado

    df_5 = asignar_avisos(df_5_ordenado)

    st.write(df_5)

    # Crear archivo Excel en memoria
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_5.to_excel(writer, index=False, sheet_name='Datos_procesados')

    output.seek(0)
    excel_bytes = output.read()

    st.download_button(
        label="Descargar archivo Excel",
        data=excel_bytes,
        file_name="datos_procesados.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.title("Mapa interactivo")

    mapa = folium.Map(location=[df_5['Latitud'][0], df_5['Longitud'][0]], zoom_start=12)

    for i, row in df_5.iterrows():
        folium.Marker(
            location=[row['Latitud'], row['Longitud']],
            popup=row['DIRECCION_COMPLETA']
        ).add_to(mapa)

    coordenadas = list(zip(df_5['Latitud'], df_5['Longitud']))

    folium.PolyLine(locations=coordenadas, color='blue', weight=2.5, opacity=1).add_to(mapa)

    st_folium(mapa, width=700, height=500)

else:
    st.write("Por favor, sube un archivo para procesar los datos")
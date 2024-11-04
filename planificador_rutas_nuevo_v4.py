import streamlit as st
import pandas as pd
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import st_folium
from geopy.adapters import RequestsAdapter
from requests.adapters import HTTPAdapter
import ssl
import numpy as np
from io import BytesIO
import re
import io


# Título de la aplicación
st.title('Planificador de rutas 3.0')

# Cargar el archivo Excel
uploaded_file = st.file_uploader("Por favor, carga un archivo Excel con las direcciones de la ruta a planificar", type=["xlsx", "xls"])

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


# Inicializar el geocodificador Nominatim (se cachea para evitar múltiples inicializaciones)
def iniciar_geolocator():
    return Nominatim(user_agent="tu_aplicacion_de_geocodificacion")

geolocator = iniciar_geolocator()

# Función para limpiar la dirección
def limpiar_direccion(direccion):
    # 1. Eliminar información de piso, departamento, o caracteres adicionales innecesarios
    # Eliminamos términos relacionados con pisos, departamentos, etc.
    direccion_limpia = re.sub(r"(piso|dpto|departamento|apartamento|#|esc|izq|der|bajo|ptal|portal(.*\s?[^\w\s,]+))", "", direccion, flags=re.IGNORECASE)

    # 2. Agregar "C/" si no está presente
    if not direccion_limpia.strip().startswith("C/"):
        direccion_limpia = "C/ " + direccion_limpia.strip()

    # 3. Eliminar caracteres no alfanuméricos, pero mantener la coma y espacio
    direccion_limpia = re.sub(r"[^a-zA-Z0-9ÁÉÍÓÚáéíóúñÑ\s,\/]", "", direccion_limpia)

    # 4. Limpiar espacios adicionales y ajustar las comas
    direccion_limpia = re.sub(r"\s+", " ", direccion_limpia)  # Reemplazar múltiples espacios por uno solo
    direccion_limpia = re.sub(r"\s*,\s*", ", ", direccion_limpia)  # Asegurar que las comas estén bien colocadas

    # 5. Usar expresión regular para extraer solo la calle, el primer número y la ciudad
    # Buscamos: "C/ CALLE, NÚMERO, CIUDAD"
    patron = re.compile(r"C/\s*([A-Za-z0-9\s]+),\s*(\d+)[^,]*,\s*([A-Za-z\s]+)$", flags=re.IGNORECASE)
    
    # 6. Si coincide con el patrón, devolver solo la calle, número y ciudad
    match = patron.match(direccion_limpia)
    if match:
        calle = match.group(1).strip()
        numero = match.group(2).strip()
        ciudad = match.group(3).strip()
        
        # 7. Retornar la dirección limpia en el formato deseado: "C/ CALLE, NÚMERO, CIUDAD"
        return f"C/ {calle}, {numero}, {ciudad}"

    # 8. Si no coincide con el formato esperado, devolver None
    print(f"Dirección {direccion_limpia} no coincide con el formato esperado.")
    return None

# Definir la función
def sustituir_ñ(texto):
    texto = str(texto) if texto is not None else ''
    # Sustituir 'ñ' por 'n' y 'Ñ' por 'N'
    texto = re.sub(r'ñ', 'n', texto)
    texto = re.sub(r'Ñ', 'N', texto)
    texto = re.sub(r'SSR', 'SAN SEBASTIÁN DE LOS REYES, MADRID', texto)
    texto = re.sub(r'AVDA', 'AVENIDA', texto)
    texto = re.sub(r'CTRA\.\s*', 'CARRETERA ', texto)
    texto = re.sub(r',\s*', ', ', texto)
    texto = re.sub(r'CASTILLA LA MANCHA', 'DE CASTILLA-LA MANCHA', texto)
    return texto



# Inicializar el geolocalizador
def iniciar_geolocator():
    return Nominatim(user_agent="tu_aplicacion_de_geocodificacion")

geolocator = iniciar_geolocator()

# Función para obtener coordenadas
def obtener_coordenadas(direccion):
    try:
        location = geolocator.geocode(direccion, country_codes='es', timeout=10)
        if location:
            return (location.latitude, location.longitude)
        else:
            return (None, None)
    except Exception as e:
        print(f"Error obteniendo coordenadas para {direccion}: {e}")
        return (None, None)

# Cálculo de la distancia con validación de None
def haversine(lat1, lon1, lat2, lon2):
    # Verificar si alguna coordenada es None
    if None in [lat1, lon1, lat2, lon2]:
        return None  # O puedes devolver np.nan, 0, o una distancia simbólica

    # Si todas las coordenadas son válidas, aplicar la fórmula de Haversine
    R = 6371  # Radio de la Tierra en km
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c



# Prcesamiento, limpieza y cara de datos

if uploaded_file is not None:
    direcciones = cargar_excel(uploaded_file)  # Carga optimizada con cache
    st.write("Archivo cargado correctamente!")


    # Transformaciones columnas de DataFrame
    df_1 = pd.DataFrame(direcciones)
    df_1['NUMERO'] = df_1['NUMERO'].fillna(0).astype(int).astype(str)
    df_3 = df_1['DIRECCION COMPLETA'] + ',' + df_1['NUMERO'].fillna(0).astype(str) + ',' + df_1['CIUDAD']   
    df_3 = pd.DataFrame(df_3, columns=['DIRECCION_COMPLETA'])
    df_4 = pd.concat([df_1, df_3], axis=1)
    df_4['AVISO'] = 'Sin Aviso'
    #df_4 = df_4.sort_values(by=['CODIGO POSTAL'])
    df_4 = df_4.drop('DIRECCION COMPLETA', axis=1)

        # Aplicar la función de limpiar_direccion a la columna DIRECCION_COMPLETA
    df_4['DIRECCION_COMPLETA'] = df_4['DIRECCION_COMPLETA'].apply(sustituir_ñ)

# Campo de entrada de texto en lugar de selectbox para que el usuario introduzca la dirección manualmente
    direccion_seleccionada = st.text_input("Introduce la dirección de punto de partida: Calle, Número, Ciudad")

    if direccion_seleccionada:
    # Filtrar el DataFrame basándose en la dirección ingresada
        df_filtrado = df_4[df_4['DIRECCION_COMPLETA'] == direccion_seleccionada]
    
    # Si la dirección no está en el DataFrame, añadirla
        if df_filtrado.empty:
            nueva_fila = pd.DataFrame({"DIRECCION_COMPLETA": [direccion_seleccionada]})
            df_4 = pd.concat([nueva_fila, df_4], ignore_index=True)
            st.write(f"Dirección '{direccion_seleccionada}' añadida a la lista de direcciones.")
        else:
            st.write(f"Datos filtrados para la dirección: {direccion_seleccionada}")
    
    # Mostrar el DataFrame actualizado o filtrado
        #st.write(df_4)
    #else:
        #st.write("Por favor, introduce una dirección para continuar.")

# Si la dirección seleccionada no está en el DataFrame, añadirla
    if direccion_seleccionada not in df_4['DIRECCION_COMPLETA'].values:
        nueva_fila = pd.DataFrame({"DIRECCION_COMPLETA": [direccion_seleccionada]})
        df_4 = pd.concat([nueva_fila, df_4], ignore_index=True)
    
# Ordenar el DataFrame para que la dirección seleccionada esté al inicio
    df_4['Orden'] = df_4['DIRECCION_COMPLETA'].apply(lambda x: 0 if x == direccion_seleccionada else 1)
    df_4 = df_4.sort_values(by='Orden').drop(columns='Orden').reset_index(drop=True)


# Aplicar la función para obtener latitud y longitud con cache
    df_4['Latitud'], df_4['Longitud'] = zip(*df_4['DIRECCION_COMPLETA'].apply(obtener_coordenadas))

    # FUCION Y ORDENAMIENTO
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

    # Crear DataFrame ordenado
        df_ordenado = df_4.loc[orden_recorrido].reset_index(drop=True)
        df_ordenado['Distancia'] = 0.0
        for i in range(1, len(df_ordenado)):
            lat_anterior, lon_anterior = df_ordenado.loc[i-1, 'Latitud'], df_ordenado.loc[i-1, 'Longitud']
            lat_actual, lon_actual = df_ordenado.loc[i, 'Latitud'], df_ordenado.loc[i, 'Longitud']
            df_ordenado.loc[i, 'Distancia'] = haversine(lat_anterior, lon_anterior, lat_actual, lon_actual)

        return df_ordenado

# Aplicar la función de ordenación al DataFrame después de obtener las coordenadas
    df_5_ordenado = ordenar_por_proximidad(df_4)

    # Asignar avosos cada 9 OT/CASO
    def asignar_avisos(df):
        pendientes = df

        # Crear una nueva columna 'AVISO' con valores NaN inicialmente
        df_5_ordenado['AVISO'] = np.nan

        # Asignar número de aviso por cada 9 OT/CASO Pendiente
        aviso_num = 1  # Comenzamos con el primer número de aviso
        for i in range(0, len(pendientes), 9):  # Saltamos de 9 en 9
            df_5_ordenado.loc[pendientes.index[i:i+9], 'AVISO'] = aviso_num
            aviso_num += 1  # Incrementar el número de aviso

        return df_5_ordenado

    # Aplicar la función para asignar los números de aviso
    df_5 = asignar_avisos(df_5_ordenado)

    # Mostrar el DataFrame actualizado con la columna 'AVISO'
    st.write(df_5)

     # Crear archivo Excel en memoria
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_5.to_excel(writer, index=False, sheet_name='Datos_procesados')

    # Convertir BytesIO a bytes para que sea descargable
    output.seek(0)
    excel_bytes = output.read()

    # Añadir botón de descarga para el archivo Excel
    st.download_button(
        label="Descargar archivo Excel",
        data=excel_bytes,
        file_name="datos_procesados.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # Título de la aplicación
    st.title("Mapa interactivo")

# Crear un mapa usando Folium 
    mapa = folium.Map(location=[df_5['Latitud'][0], df_5['Longitud'][0]], zoom_start=12)

# Añadir un marcador en una ubicación específica
    for i, row in df_5.iterrows():
        folium.Marker(
            location=[row['Latitud'], row['Longitud']],  # Coordenadas de la fila
            popup=row['DIRECCION_COMPLETA']  # Texto que aparecerá al hacer clic en el marcador
        ).add_to(mapa)

# Crear una lista de coordenadas (latitud, longitud) para la línea
    coordenadas = list(zip(df_5['Latitud'], df_5['Longitud']))


# Mostrar el número del marcador
    icon=folium.DivIcon(html=f"""<div style="font-family: sans-serif; color: red">{i+1}</div>""")

# Linea que une los marcadores
    folium.PolyLine(locations=coordenadas, color='blue', weight=2.5, opacity=1).add_to(mapa)

# Mostrar el mapa en Streamlit
    st_folium(mapa, width=700, height=500)



else:
    st.write("Por favor, sube un archivo para procesar los datos")





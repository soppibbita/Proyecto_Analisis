import pandas as pd
import numpy as np
import re
import ast
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# --- Descarga de recursos de NLTK (si no los tienes ya) ---
# He ajustado el manejo de excepciones para usar LookupError
print("Verificando y descargando recursos de NLTK...")
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    print("Recurso 'stopwords' descargado.")

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    print("Recurso 'wordnet' descargado.")

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')
    print("Recurso 'omw-1.4' descargado.")
print("Verificación y descarga de recursos de NLTK completada.")


# # 1. Preparación del dataset

# Nombre del archivo CSV. Asegúrate de que esté en la ruta 'data/dblp_stream.csv'
# Si tu archivo se llama 'dblp-2025-03-01 - 15.03.2025.csv' y está en la misma carpeta,
# cámbialo a: file_path = "dblp-2025-03-01 - 15.03.2025.csv"
file_path = "updated_dataset.csv" # Asegúrate de que esta ruta sea correcta

# Carga el DataFrame, manejando el DtypeWarning para columnas con tipos mezclados
# Puedes especificar dtypes si sabes cuáles son, o usar low_memory=False
print("Cargando el archivo CSV...")
try:
    df = pd.read_csv(file_path, low_memory=False)
    print("Archivo CSV cargado correctamente.")
except FileNotFoundError:
    print(f"Error: El archivo '{file_path}' no fue encontrado. Por favor, verifica la ruta y el nombre del archivo.")
    exit() # Sale del script si el archivo no se encuentra

# Mostrar las últimas filas para verificar (equivalente a df.tail() en el notebook)
print("\nÚltimas filas del DataFrame cargado:")
print(df.tail())

# ### Elementos únicos por tipo
print("\n--- Análisis de Elementos Únicos por Tipo y Ejemplos ---")

if 'type' in df.columns:
    unique_types = df['type'].unique()
    print(f"\nTipos de publicación únicos encontrados en df: {unique_types}")

    for pub_type in unique_types:
        print(f"\n----- Ejemplos para el Tipo: '{pub_type}' -----")
        samples = df[df['type'] == pub_type].head(5)

        if not samples.empty:
            # Asegúrate de que las columnas existan antes de intentar imprimirlas
            cols_to_print = ['title', 'booktitle', 'type', 'year']
            existing_cols = [col for col in cols_to_print if col in samples.columns]
            print(samples[existing_cols].to_string(index=False))
        else:
            print(f"No hay ejemplos disponibles para el tipo '{pub_type}'.")

    type_counts = df['type'].value_counts()
    print("\nConteo de tipos de publicación:")
    print(type_counts)
else:
    print("\nLa columna 'type' no existe en el DataFrame.")

# ### Elementos nulos en columna "year" por tipo de publicación
print("\n--- Porcentaje de Valores Nulos en 'year' por 'type' ---")

# Convertir 'year' a numérico antes de agrupar para evitar errores si contiene no-numéricos
if 'year' in df.columns:
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    if 'type' in df.columns:
        grouped_by_type = df.groupby('type')

        null_years_count = grouped_by_type['year'].apply(lambda x: x.isnull().sum())

        total_entries_count = grouped_by_type.size()

        # Evitar división por cero
        percentage_nulls_by_type = (null_years_count / total_entries_count) * 100
        percentage_nulls_by_type = percentage_nulls_by_type.fillna(0).round(2)

        print("\nConteo de Nulos en 'year' por Tipo:")
        print(null_years_count)

        print("\nTotal de Entradas por Tipo:")
        print(total_entries_count)

        print("\nPorcentaje de Valores Nulos en 'year' por Tipo:")
        print(percentage_nulls_by_type.astype(str) + '%')

        summary_nulls_by_type = pd.DataFrame({
            'Nulos_en_Year': null_years_count,
            'Total_Entradas': total_entries_count,
            'Porcentaje_Nulos_Year': percentage_nulls_by_type
        })

        print("\nResumen Completo de Nulos en 'year' por Tipo:")
        print(summary_nulls_by_type.to_string())
    else:
        print("\nLa columna 'type' no existe. No se puede analizar nulos en 'year' por tipo.")
else:
    print("\nLa columna 'year' no existe en el DataFrame original. No se puede realizar el análisis de nulos por tipo.")


# ### Analizando títulos únicos en tipo www, a ver si entrega información interesante para el estudio
tipo_a_analizar = 'www'

print(f"\n--- Análisis para el Tipo: '{tipo_a_analizar}' ---")

if 'type' in df.columns:
    # Crear una copia para evitar SettingWithCopyWarning
    df_filtered_by_type = df[df['type'] == tipo_a_analizar].copy()

    print(f"Filas iniciales para el tipo '{tipo_a_analizar}': {len(df_filtered_by_type)}")

    initial_rows_after_type_filter = len(df_filtered_by_type)

    # Asegurarse de que 'title' exista antes de usarlo
    if 'title' in df_filtered_by_type.columns:
        df_filtered_by_type.dropna(subset=['title'], inplace=True)
        dropped_nan_titles = initial_rows_after_type_filter - len(df_filtered_by_type)

        if dropped_nan_titles > 0:
            print(f"Se eliminaron {dropped_nan_titles} filas con título NaN para el tipo '{tipo_a_analizar}'.")
        else:
            print(f"No se encontraron filas con título NaN para el tipo '{tipo_a_analizar}'.")

        rows_to_drop_homepage = df_filtered_by_type[df_filtered_by_type['title'] == 'Home Page'].index
        df_filtered_by_type.drop(index=rows_to_drop_homepage, inplace=True)
        dropped_homepage_titles = len(rows_to_drop_homepage)

        if dropped_homepage_titles > 0:
            print(f"Se eliminaron {dropped_homepage_titles} filas con título 'Home Page' para el tipo '{tipo_a_analizar}'.")
        else:
            print(f"No se encontraron filas con título 'Home Page' para el tipo '{tipo_a_analizar}'.")

        print(f"\n--- Conteo de Títulos Únicos para el Tipo: '{tipo_a_analizar}' después de limpieza ---")
        if not df_filtered_by_type.empty:
            unique_titles_count = df_filtered_by_type['title'].nunique()

            print(f"Número total de publicaciones de tipo '{tipo_a_analizar}' después de limpieza: {len(df_filtered_by_type)}")
            print(f"Número de títulos únicos para el tipo '{tipo_a_analizar}': {unique_titles_count}")
            print("\nAlgunos ejemplos de títulos únicos (primeros 20 después de limpieza):")
            print(df_filtered_by_type['title'].drop_duplicates().head(20).tolist())

            # Recalcular nulos en 'year' para el tipo 'www' después de la limpieza de títulos
            if 'year' in df_filtered_by_type.columns:
                null_years_count_for_type = df_filtered_by_type['year'].isnull().sum()
                total_entries_for_type = len(df_filtered_by_type)

                if total_entries_for_type > 0:
                    percentage_nulls_for_type = (null_years_count_for_type / total_entries_for_type) * 100
                    percentage_nulls_for_type = round(percentage_nulls_for_type, 2)
                else:
                    percentage_nulls_for_type = 0.0

                print("\n--- Conteo y Porcentaje de Nulos en 'year' para el Tipo Específico (después de limpieza de títulos) ---")
                print(f"Conteo de Nulos en 'year' para '{tipo_a_analizar}': {null_years_count_for_type}")
                print(f"Porcentaje de Nulos en 'year' para '{tipo_a_analizar}': {percentage_nulls_for_type}%")
            else:
                print(f"La columna 'year' no existe en el DataFrame filtrado para el tipo '{tipo_a_analizar}'.")

        else:
            print(f"No se encontraron publicaciones de tipo '{tipo_a_analizar}' en el DataFrame 'df' después de la limpieza, o el tipo no existe.")
            print("Asegúrate de que el tipo especificado es correcto y existe en la columna 'type'.")

        print("\n--- Primeras 10 filas del DataFrame filtrado y limpiado ---")
        if not df_filtered_by_type.empty:
            print(df_filtered_by_type.head(10).to_string())
        else:
            print("El DataFrame filtrado está vacío después de la limpieza.")
    else:
        print(f"\nLa columna 'title' no existe en el DataFrame para el tipo '{tipo_a_analizar}'.")
else:
    print(f"\nLa columna 'type' no existe en el DataFrame. No se puede realizar el análisis para el tipo '{tipo_a_analizar}'.")


# Notamos que el 100% de elementos nulos en columna year son elementos del tipo www, y que de estos solo 17 elementos son validos,
# por lo tanto se pueden eliminar todos los elementos con NaN en la columna year del dataframe sin perder informacion escencial.

# ### Eliminando títulos y años nulos
# Creación de df_clean y eliminación de nulos para 'title' y 'year'
# Se crea una copia para asegurar que las operaciones no afecten el DataFrame original `df`
df_clean = df.copy()

print("\nVerificando y manejando valores nulos antes de la limpieza principal...")
print(df_clean.isnull().sum())

# Asegúrate de que 'title' y 'year' existan antes de intentar eliminar nulos
if 'title' in df_clean.columns:
    df_clean.dropna(subset=['title'], inplace=True)
    print(f"Filas después de eliminar nulos en 'title': {len(df_clean)}")
else:
    print("\nLa columna 'title' no existe para eliminar nulos.")

if 'year' in df_clean.columns:
    df_clean.dropna(subset=['year'], inplace=True)
    print(f"Filas después de eliminar nulos en 'year': {len(df_clean)}")
    # Convertir 'year' a int después de eliminar nulos, ya que pd.to_numeric con 'coerce' puede haberlos creado
    df_clean['year'] = df_clean['year'].astype(int)
else:
    print("\nLa columna 'year' no existe para eliminar nulos.")

# ### Eliminando books con posible repetición con booktitle
def clean_text_for_comparison(text):
    if pd.isna(text):
        return np.nan
    text = str(text).lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    text = text.strip()
    return text

print("\n--- Eliminando filas de tipo 'book' con títulos coincidentes ---")

if 'type' in df_clean.columns and 'title' in df_clean.columns and 'booktitle' in df_clean.columns:
    total_books_initial_count = len(df_clean[df_clean['type'] == 'book'])
    print(f"Total de elementos de tipo 'book' en el DataFrame antes de la eliminación: {total_books_initial_count}")

    books_cleaned_titles = df_clean[df_clean['type'] == 'book']['title'].apply(clean_text_for_comparison).dropna()

    all_cleaned_booktitles = df_clean['booktitle'].apply(clean_text_for_comparison).dropna()

    unique_cleaned_book_titles = books_cleaned_titles.unique()
    all_unique_cleaned_booktitles = all_cleaned_booktitles.unique()

    print(f"\nNúmero de títulos únicos de tipo 'book' (después de limpieza): {len(unique_cleaned_book_titles)}")
    print(f"Número de booktitles únicos en todo el DataFrame (después de limpieza): {len(all_unique_cleaned_booktitles)}")

    set_unique_cleaned_book_titles = set(unique_cleaned_book_titles)
    set_all_unique_cleaned_booktitles = set(all_unique_cleaned_booktitles)

    matching_titles_cleaned = list(set_unique_cleaned_book_titles.intersection(set_all_unique_cleaned_booktitles))

    # Identificar índices a eliminar
    rows_to_drop_indices = df_clean[
        (df_clean['type'] == 'book') &
        (df_clean['title'].apply(clean_text_for_comparison).isin(matching_titles_cleaned))
    ].index

    # Eliminar filas
    df_clean.drop(index=rows_to_drop_indices, inplace=True)

    dropped_count = len(rows_to_drop_indices)
    print(f"\nSe eliminaron {dropped_count} filas de tipo 'book' que tuvieron coincidencia de título.")

    total_books_after_drop = len(df_clean[df_clean['type'] == 'book'])
    print(f"Total de elementos de tipo 'book' en el DataFrame después de la eliminación: {total_books_after_drop}")

    if len(matching_titles_cleaned) > 0:
        print("\nEjemplos de los títulos limpios que causaron la coincidencia:")
        for title in matching_titles_cleaned[:10]:
            print(f"- {title}")
    else:
        print("No se encontraron coincidencias después de la limpieza de texto, por lo que no se eliminaron filas.")
else:
    print("\nColumnas 'type', 'title' o 'booktitle' no existen para realizar la limpieza de libros duplicados.")


# Eliminando columnas que no son necesarias para el análisis principal
# Asegúrate de que las columnas existan antes de intentar eliminarlas
columns_to_drop_final = ['isbn', 'editor', 'cite', 'volume', 'url', 'ee', 'crossref', 'mdate', 'pages', 'key']
existing_columns_to_drop = [col for col in columns_to_drop_final if col in df_clean.columns]
df_clean.drop(columns=existing_columns_to_drop, inplace=True, errors='ignore') # errors='ignore' para evitar errores si la columna no existe

print("\nVerificando valores nulos después de la limpieza y eliminación de columnas:")
print(df_clean.isnull().sum())
print("\nPrimeras filas del DataFrame limpio y reducido:")
print(df_clean.head())


# --- Paso 1.5: Preparación Inteligente del Contenido para Modelado de Temas ---
print("\n--- Paso 1.5 (Revisado con Limpieza Adicional de Títulos): Preparación Inteligente del Contenido para Modelado de Temas ---")

# Asegúrate de que 'year' sea int
if 'year' in df_clean.columns:
    df_clean['year'] = df_clean['year'].astype(int)

# Crear un ID único para cada publicación.
# Si 'key' fue eliminada, usamos el índice del DataFrame como base o una combinación de 'title' y 'year'
if 'key' not in df_clean.columns:
    df_clean['unique_publication_id'] = df_clean.index.astype(str)
else:
    df_clean['unique_publication_id'] = df_clean['key']

# Definir el contenido para el modelado de temas
# Inicialmente, el contenido será el título
if 'title' in df_clean.columns:
    df_clean['content_for_topic_modeling'] = df_clean['title']
else:
    df_clean['content_for_topic_modeling'] = "" # O manejar de otra forma si no hay título

# Para 'incollection', si 'booktitle' existe, usar 'booktitle' como contenido.
# Esto es crucial para agrupar artículos dentro de un mismo libro/compilación.
if 'type' in df_clean.columns and 'booktitle' in df_clean.columns:
    incollection_mask = (df_clean['type'] == 'incollection') & (df_clean['booktitle'].notna())
    df_clean.loc[incollection_mask, 'content_for_topic_modeling'] = df_clean.loc[incollection_mask, 'booktitle']

    # Actualizar ID único para incollection para que sea consistente con el booktitle y año
    df_clean.loc[incollection_mask, 'unique_publication_id'] = \
        df_clean.loc[incollection_mask, 'booktitle'] + '_' + df_clean.loc[incollection_mask, 'year'].astype(str)
else:
    print("\nAdvertencia: Las columnas 'type' o 'booktitle' no existen para la lógica de 'incollection'.")


print(f"Filas iniciales en df_clean antes de deduplicación inteligente: {len(df_clean)}")

# Columnas relevantes para el análisis final
columns_to_keep_in_analysis = [
    'type', 'authors', 'title', 'journal', 'booktitle', 'content_for_topic_modeling', 'year', 'unique_publication_id'
]
# Filtrar solo las columnas que realmente existen en df_clean
actual_columns_to_keep = [col for col in columns_to_keep_in_analysis if col in df_clean.columns]


df_for_analysis = df_clean[actual_columns_to_keep].drop_duplicates(
    subset=['unique_publication_id'], keep='first'
).copy()

print(f"Filas en df_for_analysis (después de deduplicación inteligente): {len(df_for_analysis)}")
print("\nPrimeras 10 filas de df_for_analysis (contenido y IDs):")
# Mostrar solo las columnas relevantes para la depuración
print(df_for_analysis[['type', 'title', 'booktitle', 'content_for_topic_modeling', 'year', 'unique_publication_id']].head(10).to_string())


# --- Función de limpieza de texto MEJORADA ---
def clean_text(text):
    if pd.isna(text):
        return np.nan
    text = str(text).lower()
    # Eliminar patrones comunes de números de volumen/parte/edición que pueden aparecer en títulos de libros
    text = re.sub(r'\b(?:vol(?:ume)?|part|no|number|cap|chapter|pp)\s+\d+\b', '', text)
    text = re.sub(r'\b\d+(?:st|nd|rd|th)?\s+(?:ed|edition|reprint)\b', '', text)
    # Eliminar texto entre paréntesis o corchetes que a menudo contiene metadatos
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    # Remover caracteres no alfanuméricos y dejar espacios
    text = re.sub(r'\W', ' ', text)
    # Remover espacios extra (múltiples espacios a uno solo)
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    # Remover espacios al inicio y al final
    text = text.strip()
    return text

# --- Tokenización y lematización ---
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def tokenize_and_lemmatize(text):
    if pd.isna(text):
        return np.nan
    words = text.split()
    # Filtrar palabras vacías y lematizar
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and len(word) > 2] # Eliminar palabras muy cortas
    return ' '.join(words)

# Aplicar limpieza y procesamiento de texto
# Asegurarse de que 'title' y 'booktitle' se eliminen si existían y ya no se necesitan
df_for_analysis.drop(columns=['title', 'booktitle'], inplace=True, errors='ignore')

# Aplicar las funciones de limpieza y procesamiento
df_for_analysis['cleaned_content'] = df_for_analysis['content_for_topic_modeling'].apply(clean_text)
df_for_analysis['processed_content'] = df_for_analysis['cleaned_content'].apply(tokenize_and_lemmatize)

# Eliminar filas donde processed_content sea nulo (después de la limpieza, puede haber títulos que resulten vacíos)
df_for_analysis.dropna(subset=['processed_content'], inplace=True)
# Asegurarse de que processed_content no esté vacío después de la lematización y eliminación de stopwords
df_for_analysis = df_for_analysis[df_for_analysis['processed_content'].str.strip() != '']


print(f"\nFilas en df_for_analysis (después de procesamiento de texto y eliminación de nulos/vacíos): {len(df_for_analysis)}")
print("\nPrimeras filas del DataFrame de análisis con contenido limpio y procesado:")
print(df_for_analysis[['type', 'content_for_topic_modeling', 'processed_content']].head().to_string())

# Guardar el DataFrame procesado para futuras entregas
df_for_analysis.to_csv('dblp_processed_for_analysis.csv', index=False)
print("\nDataFrame procesado guardado en 'dblp_processed_for_analysis.csv'")

# # 2. Trabajando el modelo

# --- Vectorización TF-IDF con n-gramas ---
# 'processed_content' ya está listo para ser vectorizado
tfidf_vectorizer = TfidfVectorizer(max_features=5000, min_df=5, max_df=0.85, ngram_range=(1, 3))
tfidf_matrix = tfidf_vectorizer.fit_transform(df_for_analysis['processed_content'])

print(f"\nDimensiones de la matriz TF-IDF con n-gramas: {tfidf_matrix.shape}")
print("\nPrimeras 100 características (palabras/n-gramas) detectadas:")
feature_names = tfidf_vectorizer.get_feature_names_out()
print(feature_names[:100])


# --- Modelado de Temas con Latent Dirichlet Allocation (LDA) ---
# Siguiendo la idea de los ejemplos de scikit-learn para clasificación/clustering,
# LDA es una técnica de modelado de temas que agrupa documentos en "temas"
# basados en la co-ocurrencia de palabras.

# Número de temas a descubrir. Puedes ajustar este valor.
n_components = 10 # Un buen punto de partida, puedes probar entre 5 y 20 o más.

print(f"\nEntrenando el modelo LDA con {n_components} temas...")
lda = LatentDirichletAllocation(
    n_components=n_components,
    max_iter=10,
    learning_method='online',
    random_state=42,
    n_jobs=-1 # Usa todos los núcleos disponibles
)
lda.fit(tfidf_matrix)
print("Modelo LDA entrenado.")

# Asignar el tema dominante a cada documento
doc_topic_distribution = lda.transform(tfidf_matrix)
df_for_analysis['dominant_topic'] = doc_topic_distribution.argmax(axis=1)

# Imprimir los principales términos para cada tema
def print_top_words(model, feature_names, n_top_words):
    print(f"\n--- Top {n_top_words} palabras por Tema ---")
    for topic_idx, topic in enumerate(model.components_):
        message = "Tema #%d: " % topic_idx
        message += " ".join([feature_names[i]
                              for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print("-" * 30)

n_top_words = 15 # Puedes ajustar el número de palabras por tema
print_top_words(lda, feature_names, n_top_words)

# --- Análisis de la distribución de temas ---
print("\n--- Distribución de Temas Dominantes ---")
topic_counts = df_for_analysis['dominant_topic'].value_counts().sort_index()
print(topic_counts)

# Visualización de la distribución de temas
plt.figure(figsize=(10, 6))
sns.barplot(x=topic_counts.index, y=topic_counts.values, palette='viridis')
plt.title('Distribución de Temas Dominantes')
plt.xlabel('Tema')
plt.ylabel('Número de Publicaciones')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('distribucion_temas.png') # Guarda el gráfico
plt.show()

# --- Exploración de publicaciones por tema ---
print("\n--- Ejemplos de Títulos por Tema Dominante ---")
for i in range(n_components):
    print(f"\nPublicaciones del Tema #{i}:")
    # Mostrar el título original y el año para mejor contexto
    topic_docs = df_for_analysis[df_for_analysis['dominant_topic'] == i].head(5)
    if not topic_docs.empty:
        print(topic_docs[['content_for_topic_modeling', 'year']].to_string(index=False))
    else:
        print("No hay publicaciones en este tema.")

# --- Detección de tendencias por tema a lo largo del tiempo ---
print("\n--- Tendencias de Temas a lo largo del Tiempo ---")
if 'year' in df_for_analysis.columns:
    topic_yearly_counts = df_for_analysis.groupby(['year', 'dominant_topic']).size().unstack(fill_value=0)

    plt.figure(figsize=(14, 8))
    topic_yearly_counts.plot(kind='area', stacked=True, colormap='viridis', figsize=(14, 8))
    plt.title('Tendencia de Temas a lo largo del Tiempo')
    plt.xlabel('Año')
    plt.ylabel('Número de Publicaciones')
    plt.legend(title='Tema', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('tendencias_temas_tiempo.png')
    plt.show()

    # Si quieres ver la normalización para el porcentaje
    topic_yearly_percentage = topic_yearly_counts.apply(lambda x: x / x.sum(), axis=1) * 100
    plt.figure(figsize=(14, 8))
    topic_yearly_percentage.plot(kind='area', stacked=True, colormap='viridis', figsize=(14, 8))
    plt.title('Tendencia de Temas a lo largo del Tiempo (Porcentaje)')
    plt.xlabel('Año')
    plt.ylabel('Porcentaje de Publicaciones')
    plt.legend(title='Tema', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('tendencias_temas_tiempo_porcentaje.png')
    plt.show()
else:
    print("\nLa columna 'year' no existe en el DataFrame para el análisis de tendencias temporales.")

print("\nModelado de temas y análisis de tendencias completado. Los gráficos han sido guardados.")
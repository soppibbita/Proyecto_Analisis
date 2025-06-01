import pandas as pd
import numpy as np
import re
import ast
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans # Cambiado de LatentDirichletAllocation a MiniBatchKMeans
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

# Nombre del archivo CSV. Asegúrate de que esté en la ruta correcta.
# Si tu archivo se llama 'dblp-2025-03-01 - 15.03.2025.csv' y está en la misma carpeta,
# cámbialo a: file_path = "dblp-2025-03-01 - 15.03.2025.csv"
file_path = "updated_dataset.csv" # ASEGÚRATE DE QUE ESTA RUTA SEA CORRECTA PARA TU CSV

# Carga el DataFrame, manejando el DtypeWarning para columnas con tipos mezclados
print("Cargando el archivo CSV...")
try:
    df = pd.read_csv(file_path, low_memory=False)
    print("Archivo CSV cargado correctamente.")
except FileNotFoundError:
    print(f"Error: El archivo '{file_path}' no fue encontrado. Por favor, verifica la ruta y el nombre del archivo.")
    exit() # Sale del script si el archivo no se encuentra

# Mostrar las últimas filas para verificar
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

if 'year' in df.columns:
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    if 'type' in df.columns:
        grouped_by_type = df.groupby('type')

        null_years_count = grouped_by_type['year'].apply(lambda x: x.isnull().sum())

        total_entries_count = grouped_by_type.size()

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
    df_filtered_by_type = df[df['type'] == tipo_a_analizar].copy()

    print(f"Filas iniciales para el tipo '{tipo_a_analizar}': {len(df_filtered_by_type)}")

    initial_rows_after_type_filter = len(df_filtered_by_type)

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


# ### Eliminando títulos y años nulos
df_clean = df.copy()

print("\nVerificando y manejando valores nulos antes de la limpieza principal...")
print(df_clean.isnull().sum())

if 'title' in df_clean.columns:
    df_clean.dropna(subset=['title'], inplace=True)
    print(f"Filas después de eliminar nulos en 'title': {len(df_clean)}")
else:
    print("\nLa columna 'title' no existe para eliminar nulos.")

if 'year' in df_clean.columns:
    df_clean.dropna(subset=['year'], inplace=True)
    print(f"Filas después de eliminar nulos en 'year': {len(df_clean)}")
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

    rows_to_drop_indices = df_clean[
        (df_clean['type'] == 'book') &
        (df_clean['title'].apply(clean_text_for_comparison).isin(matching_titles_cleaned))
    ].index

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
columns_to_drop_final = ['isbn', 'editor', 'cite', 'volume', 'url', 'ee', 'crossref', 'mdate', 'pages', 'key']
existing_columns_to_drop = [col for col in columns_to_drop_final if col in df_clean.columns]
df_clean.drop(columns=existing_columns_to_drop, inplace=True, errors='ignore')

print("\nVerificando valores nulos después de la limpieza y eliminación de columnas:")
print(df_clean.isnull().sum())
print("\nPrimeras filas del DataFrame limpio y reducido:")
print(df_clean.head())


# --- Paso 1.5: Preparación Inteligente del Contenido para Modelado de Temas ---
print("\n--- Paso 1.5 (Revisado con Limpieza Adicional de Títulos): Preparación Inteligente del Contenido para Modelado de Temas ---")

if 'year' in df_clean.columns:
    df_clean['year'] = df_clean['year'].astype(int)

if 'key' not in df_clean.columns:
    df_clean['unique_publication_id'] = df_clean.index.astype(str)
else:
    df_clean['unique_publication_id'] = df_clean['key']

if 'title' in df_clean.columns:
    df_clean['content_for_topic_modeling'] = df_clean['title']
else:
    df_clean['content_for_topic_modeling'] = ""

if 'type' in df_clean.columns and 'booktitle' in df_clean.columns:
    incollection_mask = (df_clean['type'] == 'incollection') & (df_clean['booktitle'].notna())
    df_clean.loc[incollection_mask, 'content_for_topic_modeling'] = df_clean.loc[incollection_mask, 'booktitle']

    df_clean.loc[incollection_mask, 'unique_publication_id'] = \
        df_clean.loc[incollection_mask, 'booktitle'] + '_' + df_clean.loc[incollection_mask, 'year'].astype(str)
else:
    print("\nAdvertencia: Las columnas 'type' o 'booktitle' no existen para la lógica de 'incollection'.")


print(f"Filas iniciales en df_clean antes de deduplicación inteligente: {len(df_clean)}")

columns_to_keep_in_analysis = [
    'type', 'authors', 'title', 'journal', 'booktitle', 'content_for_topic_modeling', 'year', 'unique_publication_id'
]
actual_columns_to_keep = [col for col in columns_to_keep_in_analysis if col in df_clean.columns]


df_for_analysis = df_clean[actual_columns_to_keep].drop_duplicates(
    subset=['unique_publication_id'], keep='first'
).copy()

print(f"Filas en df_for_analysis (después de deduplicación inteligente): {len(df_for_analysis)}")
print("\nPrimeras 10 filas de df_for_analysis (contenido y IDs):")
print(df_for_analysis[['type', 'title', 'booktitle', 'content_for_topic_modeling', 'year', 'unique_publication_id']].head(10).to_string())


# --- Función de limpieza de texto MEJORADA ---
def clean_text(text):
    if pd.isna(text):
        return np.nan
    text = str(text).lower()
    text = re.sub(r'\b(?:vol(?:ume)?|part|no|number|cap|chapter|pp)\s+\d+\b', '', text)
    text = re.sub(r'\b\d+(?:st|nd|rd|th)?\s+(?:ed|edition|reprint)\b', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    text = text.strip()
    return text

# --- Tokenización y lematización ---
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def tokenize_and_lemmatize(text):
    if pd.isna(text):
        return np.nan
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and len(word) > 2]
    return ' '.join(words)

df_for_analysis.drop(columns=['title', 'booktitle'], inplace=True, errors='ignore')

df_for_analysis['cleaned_content'] = df_for_analysis['content_for_topic_modeling'].apply(clean_text)
df_for_analysis['processed_content'] = df_for_analysis['cleaned_content'].apply(tokenize_and_lemmatize)

df_for_analysis.dropna(subset=['processed_content'], inplace=True)
df_for_analysis = df_for_analysis[df_for_analysis['processed_content'].str.strip() != '']


print(f"\nFilas en df_for_analysis (después de procesamiento de texto y eliminación de nulos/vacíos): {len(df_for_analysis)}")
print("\nPrimeras filas del DataFrame de análisis con contenido limpio y procesado:")
print(df_for_analysis[['type', 'content_for_topic_modeling', 'processed_content']].head().to_string())

# Guardar el DataFrame procesado para futuras entregas
df_for_analysis.to_csv('dblp_processed_for_analysis.csv', index=False)
print("\nDataFrame procesado guardado en 'dblp_processed_for_analysis.csv'")

# # 2. Trabajando el modelo (Clustering con MiniBatchKMeans)

# --- Vectorización TF-IDF con n-gramas ---
tfidf_vectorizer = TfidfVectorizer(max_features=5000, min_df=5, max_df=0.85, ngram_range=(1, 3))
tfidf_matrix = tfidf_vectorizer.fit_transform(df_for_analysis['processed_content'])

print(f"\nDimensiones de la matriz TF-IDF con n-gramas: {tfidf_matrix.shape}")
print("\nPrimeras 100 características (palabras/n-gramas) detectadas:")
feature_names = tfidf_vectorizer.get_feature_names_out()
print(feature_names[:100])


# --- Modelo de Clustering con MiniBatchKMeans ---

# --- NOTA SOBRE CLUSTERS DESEQUILIBRADOS EN K-MEANS ---
# K-Means (y MiniBatchKMeans) busca minimizar la suma de las distancias cuadradas dentro de cada cluster (inercia).
# Este objetivo NO garantiza que los clusters tengan un tamaño similar. Es muy común tener clusters con muchos
# elementos y otros con pocos, especialmente si hay temas dominantes o nichos muy específicos en los datos.
# Un cluster con 7 millones de elementos sugiere que una gran porción de tus documentos son muy similares entre sí,
# o están centrados en un tema extremadamente prevalente, y MiniBatchKMeans los agrupa eficientemente.

# --- Estrategias para obtener "clusters más uniformes" (o más representativos) ---
# 1. Ajustar 'n_clusters': Experimenta con diferentes números de clusters. Si un cluster es muy grande,
#    podrías intentar aumentar 'n_clusters' para forzar al algoritmo a subdividirlo, pero no hay garantía.
#    Por ejemplo, si tienes 10 temas y uno es gigante, intenta 20 o 30.
# 2. Ajustar los parámetros de TF-IDF:
#    - `max_df`: Aumentar `max_df` puede incluir palabras muy comunes que son ruido, pero reducirlo (ej. 0.7, 0.6)
#      podría eliminar palabras que aparecen en casi todos los documentos y no ayudan a diferenciar temas.
#      Si un cluster es gigante, podría ser que muchas palabras "discriminatorias" estén siendo filtradas.
#    - `min_df`: Aumentar `min_df` elimina palabras muy raras. Si es demasiado bajo, el modelo puede intentar
#      crear clusters basados en ruidos o palabras muy específicas.
#    - `max_features`: Limitar las características puede obligar al modelo a usar las más importantes, pero si es
#      demasiado bajo, podría perder información importante.
# 3. Métodos Avanzados (para futuro):
#    - **Clustering Jerárquico (AgglomerativeClustering):** Permite construir una jerarquía y cortar el árbol
#      en un punto que te dé un mejor balance, pero es menos escalable.
#    - **Clustering Restringido/Constrained K-Means:** Versiones modificadas de K-Means que permiten imponer
#      restricciones de tamaño o balance. No están directamente en scikit-learn estándar.
#    - **Dividir el cluster grande:** Una estrategia es identificar el cluster gigante, extraer solo esos documentos
#      y luego aplicar un nuevo proceso de clustering (ej. otro MiniBatchKMeans) solo sobre ellos para subdividirlos.
#      Esto puede ser un enfoque práctico.

# Número de categorías (clusters) a descubrir. Ajusta este valor.
num_clusters = 10 # Empieza con 10, pero ¡experimenta mucho aquí!

print(f"\nEntrenando el modelo MiniBatchKMeans con {num_clusters} categorías (clusters)...")
kmeans_model = MiniBatchKMeans(
    n_clusters=num_clusters,
    init='k-means++',      # Inicialización inteligente
    n_init='auto',         # Número de veces que el algoritmo k-means se ejecutará con diferentes centroides
    random_state=42,       # Para reproducibilidad
    batch_size=256,        # Tamaño del minibatch
    compute_labels=True,   # Calcular etiquetas al final
    max_iter=100           # Número máximo de iteraciones para un solo pase.
)

kmeans_model.fit(tfidf_matrix)
print("Modelo MiniBatchKMeans entrenado.")

df_for_analysis['cluster'] = kmeans_model.labels_

print(f"\nSe asignaron {num_clusters} categorías (clusters) a las publicaciones.")
print("\nConteo de publicaciones por categoría:")
print(df_for_analysis['cluster'].value_counts().sort_index())

print("\nPrimeras filas del DataFrame con la asignación de categorías:")
print(df_for_analysis[['processed_content', 'cluster']].head().to_string())

# --- Visualización de la Distribución de Clusters ---
plt.figure(figsize=(10, 6))
sns.countplot(x='cluster', data=df_for_analysis, palette='viridis')
plt.title('Distribución de Publicaciones por Categoría (Cluster)')
plt.xlabel('Categoría (Cluster ID)')
plt.ylabel('Número de Publicaciones')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('distribucion_clusters.png') # Guarda el gráfico
plt.show()

# --- Obtener las palabras más representativas de cada cluster ---
# Los centros de los clusters son los vectores que representan cada cluster.
# Podemos usar el TF-IDF inverso para obtener las palabras más importantes.
print("\n--- Palabras más representativas por Categoría (Cluster) ---")
order_centroids = kmeans_model.cluster_centers_.argsort()[:, ::-1]
terms = tfidf_vectorizer.get_feature_names_out()

for i in range(num_clusters):
    print(f"\nCluster {i}:")
    top_terms = [terms[ind] for ind in order_centroids[i, :15]] # Top 15 palabras por cluster
    print(f"  {', '.join(top_terms)}")

# --- Ejemplos de Títulos por Cluster ---
print("\n--- Ejemplos de Títulos por Categoría (Cluster) ---")
for i in range(num_clusters):
    print(f"\nPublicaciones del Cluster #{i}:")
    # Mostrar el título original y el año para mejor contexto
    cluster_docs = df_for_analysis[df_for_analysis['cluster'] == i].head(5)
    if not cluster_docs.empty:
        print(cluster_docs[['content_for_topic_modeling', 'year']].to_string(index=False))
    else:
        print("No hay publicaciones en este cluster.")

# --- Detección de tendencias de Clusters a lo largo del tiempo ---
print("\n--- Tendencias de Categorías (Clusters) a lo largo del Tiempo ---")
if 'year' in df_for_analysis.columns:
    cluster_yearly_counts = df_for_analysis.groupby(['year', 'cluster']).size().unstack(fill_value=0)

    plt.figure(figsize=(14, 8))
    cluster_yearly_counts.plot(kind='area', stacked=True, colormap='viridis', figsize=(14, 8))
    plt.title('Tendencia de Categorías (Clusters) a lo largo del Tiempo')
    plt.xlabel('Año')
    plt.ylabel('Número de Publicaciones')
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('tendencias_clusters_tiempo.png')
    plt.show()

    # Si quieres ver la normalización para el porcentaje
    topic_yearly_percentage = cluster_yearly_counts.apply(lambda x: x / x.sum(), axis=1) * 100
    plt.figure(figsize=(14, 8))
    topic_yearly_percentage.plot(kind='area', stacked=True, colormap='viridis', figsize=(14, 8))
    plt.title('Tendencia de Categorías (Clusters) a lo largo del Tiempo (Porcentaje)')
    plt.xlabel('Año')
    plt.ylabel('Porcentaje de Publicaciones')
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('tendencias_clusters_tiempo_porcentaje.png')
    plt.show()
else:
    print("\nLa columna 'year' no existe en el DataFrame para el análisis de tendencias temporales.")

print("\nModelado de clusters y análisis de tendencias completado. Los gráficos han sido guardados.")
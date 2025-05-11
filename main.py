import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import ast

# Nombre del archivo CSV
file_path = 'dblp-2025-03-01 - 15.03.2025.csv'

# 1. Carga de Datos con Pandas
try:
    data = pd.read_csv(file_path)
    print("Archivo CSV cargado correctamente.")
    print("\nPrimeras filas del DataFrame:")
    print(data.head())
    print("\nInformación del DataFrame:")
    print(data.info())
except FileNotFoundError:
    print(f"Error: El archivo '{file_path}' no fue encontrado.")
    exit()

# 2. Limpieza y Preparación de Datos

# a) Manejo de Valores Nulos
print("\nCantidad de valores nulos por columna:")
print(data.isnull().sum())

# Puedes decidir cómo manejar los valores nulos. Por ejemplo, eliminar filas con nulos:
# data = data.dropna()
# O rellenarlos con un valor específico:
# data['journal'] = data['journal'].fillna('Sin información')

# b) Conversión de Tipos de Datos (ejemplo para la columna 'year' si existe)
if 'year' in data.columns:
    try:
        data['year'] = pd.to_numeric(data['year'], errors='coerce').dropna().astype(int)
        print("\nColumna 'year' convertida a entero (ignorando valores no numéricos).")
    except KeyError:
        print("\nLa columna 'year' no existe.")

# c) Procesamiento de la Columna 'author'
def extract_author_names(author_list_str):
    try:
        author_list = ast.literal_eval(author_list_str)
        return [author['name'] for author in author_list]
    except (ValueError, TypeError):
        return []  # Retorna una lista vacía si hay un error al parsear

if 'author' in data.columns:
    data['authors'] = data['author'].apply(extract_author_names)
    print("\nColumna 'author' procesada y nombres de autores extraídos en la columna 'authors'.")
    print("Primeras filas de la columna 'authors':")
    print(data['authors'].head())
else:
    print("\nLa columna 'author' no existe.")

# 3. Análisis Exploratorio

# a) Análisis de la Producción Científica por Año
if 'year' in data.columns:
    yearly_counts = data['year'].value_counts().sort_index()
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=yearly_counts.index, y=yearly_counts.values)
    plt.title('Número de Publicaciones por Año')
    plt.xlabel('Año')
    plt.ylabel('Número de Publicaciones')
    plt.grid(True)
    plt.savefig('publicaciones_por_anio.png') # Guarda el gráfico
    plt.show()
else:
    print("\nNo se puede realizar el análisis de publicaciones por año porque la columna 'year' no existe.")

# b) Análisis de Frecuencia de Autores
if 'authors' in data.columns:
    all_authors = [author for sublist in data['authors'] for author in sublist]
    author_counts = Counter(all_authors)
    top_authors = author_counts.most_common(20)

    plt.figure(figsize=(10, 8))
    sns.barplot(x=[count for word, count in top_authors], y=[word for word, count in top_authors])
    plt.title('Top 20 Autores')
    plt.xlabel('Frecuencia')
    plt.ylabel('Autor')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('top_autores.png') # Guarda el gráfico
    plt.show()
else:
    print("\nNo se puede realizar el análisis de frecuencia de autores porque la columna 'authors' no existe.")

# c) Análisis de Publicaciones por Revista (journal)
if 'journal' in data.columns:
    journal_counts = data['journal'].value_counts().nlargest(20)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=journal_counts.values, y=journal_counts.index, orient='h')
    plt.title('Top 20 Revistas con Más Publicaciones')
    plt.xlabel('Número de Publicaciones')
    plt.ylabel('Revista')
    plt.tight_layout()
    plt.savefig('publicaciones_por_revista.png') # Guarda el gráfico
    plt.show()
else:
    print("\nNo se puede realizar el análisis de publicaciones por revista porque la columna 'journal' no existe.")

print("\nAnálisis exploratorio básico completado. Los gráficos han sido guardados como archivos PNG.")
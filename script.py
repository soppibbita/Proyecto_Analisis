import json
from lxml import etree
from tqdm import tqdm
import csv

# Definición de tags de interés
record_tags = {
    'article', 'inproceedings', 'proceedings',
    'book', 'incollection', 'phdthesis',
    'mastersthesis', 'www'
}

# Lista de columnas a exportar
fieldnames = [
    'type', 'key', 'mdate', 'authors', 'title', 'year',
    'journal', 'booktitle', 'pages', 'volume', 'url',
    'ee', 'crossref', 'isbn', 'editor', 'cite'
]

# Archivo de entrada y salida
xml_file = 'data/dblp.xml'
csv_file = 'dblp_stream.csv'

with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # Iterparse con recuperación de errores y barra de progreso
    context = etree.iterparse(xml_file, events=('end',), recover=True)
    for _, elem in tqdm(context, desc='Procesando registros'):
        tag = etree.QName(elem).localname
        if tag in record_tags:
            row = {
                'type': tag,
                'key': elem.get('key', ''),
                'mdate': elem.get('mdate', '')
            }

            # Autores como JSON: lista de objetos con name y orcid
            authors = []
            for a in elem.findall('author'):
                authors.append({
                    'name': a.text or '',
                    'orcid': a.get('orcid', '')
                })
            row['authors'] = json.dumps(authors, ensure_ascii=False)

            # Campos básicos
            row['title']  = (elem.findtext('title') or '')
            row['year']   = (elem.findtext('year')  or '')
            row['journal']    = (elem.findtext('journal')    or '')
            row['booktitle']  = (elem.findtext('booktitle')  or '')
            row['pages']      = (elem.findtext('pages')      or '')
            row['volume']     = (elem.findtext('volume')     or '')
            row['url']        = (elem.findtext('url')        or '')

            # Nuevos campos adicionales
            row['ee']       = (elem.findtext('ee')       or '')
            row['crossref'] = (elem.findtext('crossref') or '')
            row['isbn']     = (elem.findtext('isbn')     or '')
            row['editor']   = (elem.findtext('editor')   or '')

            # Campo cite (varios posibles)
            cites = [c.text for c in elem.findall('cite') if c.text]
            row['cite'] = '|'.join(cites)

            # Escribe la fila en el CSV
            writer.writerow(row)

            # Limpieza para no acumular el árbol en memoria
            parent = elem.getparent()
            elem.clear()
            if parent is not None:
                parent.remove(elem)

    del context

print(f"Procesamiento completado. Archivo generado: {csv_file}")

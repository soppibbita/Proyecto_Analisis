import sys
import csv
import json
from lxml import etree
from tqdm import tqdm

def contar_lineas(archivo):
    with open(archivo, 'rb') as f:
        return sum(1 for _ in f)

def procesar_y_guardar_xml_grande(archivo_xml, archivo_csv):
    print(f"Procesando XML: {archivo_xml} y escribiendo directamente en {archivo_csv}")
    
    campos_comunes = [
        'title', 'pages', 'year', 'booktitle', 'journal', 'volume', 'number',
        'school', 'crossref', 'publisher', 'isbn', 'series', 'editor'
    ]

    campos_detectados = set([
        'tipo', 'key', 'mdate', 'publtype',
        'autores', 'atributos_autores',
        'enlaces', 'atributos_ee', 'urls'
    ])

    # Paso 1: detectar campos adicionales en un muestreo de entradas
    print("Detectando campos...")
    context = etree.iterparse(archivo_xml, events=("end",), encoding='ISO-8859-1', recover=True)
    count = 0
    for event, elem in context:
        if elem.tag == 'dblp':
            continue
        for child in elem:
            campos_detectados.add(child.tag)
            if child.attrib:
                campos_detectados.add(f"{child.tag}_attrs")
        count += 1
        if count > 1000:
            break
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]
    del context

    campos_ordenados = sorted(campos_detectados)
    if 'tipo' in campos_ordenados:
        campos_ordenados.remove('tipo')
        campos_ordenados = ['tipo'] + campos_ordenados

    print("Procesando archivo completo y escribiendo CSV...")
    total_lineas = contar_lineas(archivo_xml)

    with open(archivo_csv, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.DictWriter(f_out, fieldnames=campos_ordenados)
        writer.writeheader()

        context = etree.iterparse(archivo_xml, events=("end",), encoding='ISO-8859-1', recover=True)
        for event, elem in tqdm(context, total=total_lineas, desc="Procesando", unit="líneas", smoothing=0.1):
            if elem.tag == 'dblp':
                continue

            tipo = elem.tag
            entrada = {
                'tipo': tipo,
                'key': elem.attrib.get('key', ''),
                'mdate': elem.attrib.get('mdate', ''),
                'publtype': elem.attrib.get('publtype', '')
            }

            for campo in campos_comunes:
                sub = elem.find(campo)
                entrada[campo] = sub.text.strip() if sub is not None and sub.text else ''

            autores = elem.findall('author')
            entrada['autores'] = '|'.join([a.text.strip() for a in autores if a.text])
            atributos_autores = [json.dumps({a.text.strip(): a.attrib}) for a in autores if a.attrib and a.text]
            if atributos_autores:
                entrada['atributos_autores'] = '|'.join(atributos_autores)

            ees = elem.findall('ee')
            entrada['enlaces'] = '|'.join([e.text.strip() for e in ees if e.text])
            atributos_ee = [json.dumps({e.text.strip(): e.attrib}) for e in ees if e.attrib and e.text]
            if atributos_ee:
                entrada['atributos_ee'] = '|'.join(atributos_ee)

            urls = elem.findall('url')
            entrada['urls'] = '|'.join([u.text.strip() for u in urls if u.text])

            for sub in elem:
                if sub.tag in campos_comunes + ['author', 'ee', 'url']:
                    continue
                if sub.tag not in entrada:
                    entrada[sub.tag] = sub.text.strip() if sub.text else ''
                    if sub.attrib:
                        entrada[f"{sub.tag}_attrs"] = json.dumps(sub.attrib)

            entrada_filtrada = {k: entrada.get(k, '') for k in campos_ordenados}
            writer.writerow(entrada_filtrada)

            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]

    print(f"\n✅ Archivo procesado correctamente y guardado en: {archivo_csv}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python procesar_xml_grande.py <archivo.xml> <salida.csv>")
        sys.exit(1)

    archivo_xml = sys.argv[1]
    archivo_csv = sys.argv[2]

    try:
        procesar_y_guardar_xml_grande(archivo_xml, archivo_csv)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

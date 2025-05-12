from lxml import etree
import pandas as pd
import csv
import sys
import json
from collections import defaultdict

def procesar_xml(archivo_xml):
    """
    Procesa un archivo XML grande de forma eficiente y extrae datos por entrada.
    """
    print(f"Procesando archivo XML grande: {archivo_xml}")
    
    datos = []
    tipos = set()

    campos_comunes = [
        'title', 'pages', 'year', 'booktitle', 'journal', 'volume', 'number',
        'school', 'crossref', 'publisher', 'isbn', 'series', 'editor'
    ]

    contexto = etree.iterparse(archivo_xml, events=("end",), tag=None, encoding='ISO-8859-1')

    for evento, elem in contexto:
        if elem.tag == 'dblp':
            continue  # No procesar el contenedor principal

        tipo = elem.tag
        tipos.add(tipo)

        entrada_dict = {
            'tipo': tipo,
            'key': elem.attrib.get('key', ''),
            'mdate': elem.attrib.get('mdate', ''),
            'publtype': elem.attrib.get('publtype', '')
        }

        for campo in campos_comunes:
            encontrado = elem.find(campo)
            entrada_dict[campo] = encontrado.text.strip() if encontrado is not None and encontrado.text else ''

        autores = elem.findall('author')
        entrada_dict['autores'] = '|'.join([a.text.strip() for a in autores if a.text]) if autores else ''

        atributos_autores = []
        for autor in autores:
            attr = autor.attrib
            if attr:
                atributos_autores.append(json.dumps({autor.text.strip(): attr}))
        if atributos_autores:
            entrada_dict['atributos_autores'] = '|'.join(atributos_autores)

        ees = elem.findall('ee')
        entrada_dict['enlaces'] = '|'.join([e.text.strip() for e in ees if e.text]) if ees else ''

        atributos_ee = []
        for ee in ees:
            attr = ee.attrib
            if attr:
                atributos_ee.append(json.dumps({ee.text.strip(): attr}))
        if atributos_ee:
            entrada_dict['atributos_ee'] = '|'.join(atributos_ee)

        urls = elem.findall('url')
        entrada_dict['urls'] = '|'.join([u.text.strip() for u in urls if u.text]) if urls else ''

        for sub in elem:
            if sub.tag in campos_comunes + ['author', 'ee', 'url']:
                continue
            if sub.tag not in entrada_dict:
                entrada_dict[sub.tag] = sub.text.strip() if sub.text else ''
                if sub.attrib:
                    entrada_dict[f"{sub.tag}_attrs"] = json.dumps(sub.attrib)
            else:
                # Si ya existe, crear un nombre alternativo
                i = 1
                while f"{sub.tag}_{i}" in entrada_dict:
                    i += 1
                entrada_dict[f"{sub.tag}_{i}"] = sub.text.strip() if sub.text else ''
                if sub.attrib:
                    entrada_dict[f"{sub.tag}_{i}_attrs"] = json.dumps(sub.attrib)

        datos.append(entrada_dict)

        # Liberar memoria
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]

    return datos, tipos


def guardar_csv(datos, archivo_salida):
    """
    Guarda los datos en un archivo CSV
    """
    if not datos:
        print("No hay datos para guardar.")
        return
    
    # Obtener todos los campos únicos
    todos_campos = set()
    for entrada in datos:
        todos_campos.update(entrada.keys())
    
    # Ordenar campos para tener un orden consistente
    campos_ordenados = sorted(todos_campos)
    
    # Mover 'tipo' al principio si existe
    if 'tipo' in campos_ordenados:
        campos_ordenados.remove('tipo')
        campos_ordenados = ['tipo'] + campos_ordenados
    
    print(f"Guardando {len(datos)} entradas en {archivo_salida}")
    with open(archivo_salida, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=campos_ordenados)
        writer.writeheader()
        writer.writerows(datos)
    
    print(f"Archivo CSV guardado exitosamente: {archivo_salida}")

def guardar_por_tipo(datos, prefijo_salida):
    """
    Guarda los datos en archivos CSV separados por tipo
    """
    # Agrupar datos por tipo
    datos_por_tipo = defaultdict(list)
    for entrada in datos:
        tipo = entrada['tipo']
        datos_por_tipo[tipo].append(entrada)
    
    # Guardar cada tipo en un archivo separado
    for tipo, entradas in datos_por_tipo.items():
        archivo_salida = f"{prefijo_salida}_{tipo}.csv"
        guardar_csv(entradas, archivo_salida)

def main():
    if len(sys.argv) < 2:
        print("Uso: python procesar_xml_dblp.py <archivo_xml> [archivo_salida]")
        sys.exit(1)
    
    archivo_xml = sys.argv[1]
    archivo_salida = sys.argv[2] if len(sys.argv) > 2 else "dblp_procesado.csv"
    
    try:
        datos, tipos = procesar_xml(archivo_xml)
        guardar_csv(datos, archivo_salida)
        
        # También guardar archivos separados por tipo
        prefijo = archivo_salida.rsplit('.', 1)[0]
        guardar_por_tipo(datos, prefijo)
        
        print("\nResumen:")
        print(f"Total de entradas procesadas: {len(datos)}")
        print(f"Tipos encontrados: {tipos}")
        
        # Mostrar conteo por tipo
        conteo_por_tipo = defaultdict(int)
        for entrada in datos:
            conteo_por_tipo[entrada['tipo']] += 1
        
        print("\nConteo por tipo:")
        for tipo, conteo in conteo_por_tipo.items():
            print(f"  - {tipo}: {conteo} entradas")
            
    except Exception as e:
        print(f"Error al procesar el archivo: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
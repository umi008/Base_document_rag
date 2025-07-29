import os
import re
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
import easyocr
import numpy as np
from pdf2image import convert_from_path

def limpiar_texto(texto: str) -> str:
    """
    Limpia y normaliza el texto de entrada.
    - Elimina espacios en blanco, saltos de línea y tabulaciones extra.
    - Convierte todo el texto a minúsculas.
    - Elimina caracteres especiales no relevantes y artefactos de formato.
    """
    texto = re.sub(r'\s+', ' ', texto)
    texto = texto.lower()
    texto = re.sub(r'[^a-z0-9áéíóúñü.,;¿?¡! ]', '', texto)
    texto = texto.strip()
    return texto

def cargar_documentos(data_dir="data/"):
    """
    Carga documentos de diferentes formatos desde un directorio y los convierte a texto plano (Markdown básico).
    Soporta archivos .txt, .docx y .pdf.

    Args:
        data_dir (str): Ruta al directorio donde se encuentran los archivos.

    Returns:
        list: Lista de documentos cargados como objetos LangChain Document.
    """
    documentos = []
    reader = easyocr.Reader(['es', 'en'])
    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        if filename.lower().endswith(".pdf"):
            print(f"Cargando archivo PDF: {filename}")
            loader = PyPDFLoader(filepath)
            docs = loader.load()
            # Si el PDF no tiene texto extraíble, usar OCR
            for doc in docs:
                contenido = doc.page_content or ""
                if len(contenido.strip()) < 20:
                    # OCR para PDFs escaneados
                    print(f"Aplicando OCR a {filename}...")
                    imagenes = convert_from_path(filepath)
                    texto_extraido_lista = []
                    for imagen in imagenes:
                        # Convertir la imagen de PIL a un array de numpy para easyocr
                        resultado = reader.readtext(np.array(imagen))
                        # Unir el texto detectado en la página
                        texto_pagina = " ".join([res[1] for res in resultado])
                        texto_extraido_lista.append(texto_pagina)
                    texto_extraido = "\n".join(texto_extraido_lista)
                    doc.page_content = texto_extraido
            documentos.extend(docs)
        elif filename.lower().endswith(".docx"):
            print(f"Cargando archivo DOCX: {filename}")
            loader = Docx2txtLoader(filepath)
            documentos.extend(loader.load())
        elif filename.lower().endswith(".txt"):
            print(f"Cargando archivo de texto: {filename}")
            loader = TextLoader(filepath, encoding="utf-8")
            documentos.extend(loader.load())
    # Limpiar el contenido de cada documento
    for doc in documentos:
        doc.page_content = limpiar_texto(doc.page_content)
    return documentos
import os
import re
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader

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
    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        if filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(filepath)
            documentos.extend(loader.load())
        elif filename.lower().endswith(".docx"):
            loader = Docx2txtLoader(filepath)
            documentos.extend(loader.load())
        elif filename.lower().endswith(".txt"):
            loader = TextLoader(filepath, encoding="utf-8")
            documentos.extend(loader.load())
    # Limpiar el contenido de cada documento
    for doc in documentos:
        doc.page_content = limpiar_texto(doc.page_content)
    return documentos
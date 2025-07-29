import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.data_loader import cargar_documentos

def crear_vector_store(data_dir="data/", persist_directory="db", rebuild=False):
    """
    Carga documentos (ya limpios), los divide en fragmentos, genera embeddings y los almacena en ChromaDB.
    Si rebuild=True o se detectan cambios, elimina la base de datos anterior antes de crear la nueva.
    """
    # Cargar variables de entorno
    load_dotenv(
        dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'),
        verbose=True
    )
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("La variable de entorno GOOGLE_API_KEY no está configurada.")

    # Eliminar base de datos anterior si se solicita reconstrucción
    if rebuild and os.path.exists(persist_directory):
        import shutil
        shutil.rmtree(persist_directory)
        print(f"Directorio '{persist_directory}' eliminado para reconstrucción.")

    # 1. Cargar documentos (ya limpios)
    documentos = cargar_documentos(data_dir)

    # 2. Dividir en fragmentos y enriquecer metadatos
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    fragmentos = []
    for doc in documentos:
        nombre_archivo = doc.metadata.get("source", "desconocido") if hasattr(doc, "metadata") else "desconocido"
        fragmentos += text_splitter.create_documents(
            [doc.page_content],
            metadatas=[{"source": nombre_archivo}]
        )

    # 3. Inicializar embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # 4. Crear y persistir el VectorStore
    vector_store = Chroma.from_documents(
        documents=fragmentos,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    vector_store.persist()
    print(f"Vector store creado en '{persist_directory}' con {len(fragmentos)} fragmentos.")
    return vector_store
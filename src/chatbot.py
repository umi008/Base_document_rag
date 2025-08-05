import os
import langchain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

# Cargar el system prompt desde archivo
def cargar_system_prompt(path="system_prompt.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# Almacén en memoria para las historias de chat
store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    """Obtiene o crea un historial de chat para un ID de sesión."""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Prompt actualizado para usar con LCEL y MessagesPlaceholder
custom_human_prompt = """
<CONTEXT>
{context}
</CONTEXT>

<QUESTION>
{question}
</QUESTION>
"""

def crear_cadena_conversacional(persist_directory="db"):
    """
    Crea la cadena de recuperación conversacional usando LCEL y RunnableWithMessageHistory.
    """

    # Cargar variables de entorno (incluye GOOGLE_API_KEY)
    load_dotenv()
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("La variable de entorno GOOGLE_API_KEY no está definida.")

    # Inicializar embeddings y cargar VectorStore persistente
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type="retrieval_query")
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    # Inicializar el modelo de lenguaje Gemini
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.8)

    # Cargar el system prompt
    system_prompt_content = cargar_system_prompt()

    # Crear el ChatPromptTemplate con placeholders para historial y contexto
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_content),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", custom_human_prompt),
    ])

    def format_docs(docs):
        """Formatea los documentos recuperados en una sola cadena."""
        return "\n\n".join(doc.page_content for doc in docs)

    # Construir la cadena RAG principal con LCEL
    rag_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    # Envolver la cadena con manejo de historial de mensajes
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

    return conversational_rag_chain
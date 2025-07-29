import os
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Cargar el system prompt desde archivo
def cargar_system_prompt(path="system_prompt.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# Nuevo prompt que incluye context, chat_history y question en el orden requerido
custom_human_prompt = """
<CONTEXT>
{context}
</CONTEXT>

<CHAT_HISTORY>
{chat_history}
</CHAT_HISTORY>

<QUESTION>
{question}
</QUESTION>
"""

def crear_cadena_conversacional(persist_directory="db"):
    """
    Crea la cadena de recuperación conversacional.
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
        search_kwargs={
            "k": 5,
            #"lambda_mult": 0.5,
            #"score_threshold": 0.6
        }
    )

    # Configurar memoria conversacional
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # Inicializar el modelo de lenguaje Gemini
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.8)

    # Cargar el system prompt
    system_prompt_content = cargar_system_prompt()

    # Crear el ChatPromptTemplate con mensaje de sistema y humano
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_content),
        ("human", custom_human_prompt)
    ])

    # Crear la cadena conversacional
    cadena = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=True,
        combine_docs_chain_kwargs={"prompt": prompt}
    )
    return cadena
import os
from dotenv import load_dotenv
from src.vector_store import crear_vector_store
from src.chatbot import crear_cadena_conversacional

def main():
    load_dotenv()

    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: La variable de entorno GOOGLE_API_KEY no está configurada.")
        return

    # Crear o actualizar la base de datos vectorial
    crear_vector_store(data_dir="data/", persist_directory="db", rebuild=False)
    print("Indexación completada.")

    # Crear la cadena conversacional
    cadena_conversacional = crear_cadena_conversacional()

    print("Chatbot RAG iniciado. Escribe 'salir' para terminar.")

    # Usar un ID de sesión fijo para la conversación en la terminal
    session_id = "terminal_session"

    while True:
        pregunta = input("Tú: ")
        if pregunta.lower() == "salir":
            break

        # Invocar la cadena con el ID de sesión en la configuración
        config = {"configurable": {"session_id": session_id}}
        respuesta = cadena_conversacional.invoke({"question": pregunta}, config=config)
        
        # La nueva cadena devuelve directamente la respuesta como un string
        print(f"Bot: {respuesta}")

if __name__ == "__main__":
    main()
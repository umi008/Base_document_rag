# Chatbot RAG con Google Gemini

## Descripción General

Este proyecto implementa un chatbot conversacional avanzado basado en la arquitectura RAG (Retrieval-Augmented Generation). Su principal función es responder preguntas de los usuarios utilizando una base de conocimiento construida a partir de un conjunto de documentos proporcionados en formatos como PDF, DOCX y TXT.

El sistema procesa y extrae texto de estos documentos, incluyendo la capacidad de manejar PDFs escaneados mediante tecnología OCR. La información extraída se utiliza para construir una base de datos vectorial. Posteriormente, el chatbot emplea un Modelo de Lenguaje Grande (LLM) de Google, específicamente Gemini, para generar respuestas coherentes y contextualmente relevantes, basadas en la información recuperada de los documentos.

## Características Principales

*   **Procesamiento de Múltiples Formatos**: Capaz de leer y extraer texto de archivos PDF, DOCX y TXT.
*   **Soporte para OCR**: Integra OCR para digitalizar y procesar texto de documentos PDF escaneados.
*   **Generación Aumentada por Recuperación (RAG)**: Utiliza una base de datos vectorial (ChromaDB) para encontrar la información más relevante y un LLM (Google Gemini) para generar respuestas fluidas y precisas.
*   **Memoria Conversacional**: Mantiene el contexto de la conversación para permitir un diálogo coherente y seguir el hilo de las preguntas.
*   **Fácil de Usar**: Interfaz de línea de comandos simple para interactuar con el chatbot.

## Cómo Funciona

El flujo de trabajo del chatbot se organiza en los siguientes pasos:

1.  **Inicialización**: El script principal [`main.py`](main.py) carga las variables de entorno (como la API key de Google) y coordina la ejecución de todos los componentes del sistema.
2.  **Carga de Datos**: El módulo [`src/data_loader.py`](src/data_loader.py) se encarga de leer los archivos ubicados en el directorio `data/`. Extrae el texto de cada documento y, si es necesario, aplica OCR a los archivos PDF escaneados.
3.  **Indexación Vectorial**: A través de [`src/vector_store.py`](src/vector_store.py), el texto extraído se divide en fragmentos más pequeños (chunks). Luego, se generan embeddings (representaciones vectoriales) para cada fragmento utilizando un modelo de Google y se almacenan en una base de datos vectorial local ChromaDB, guardada en el directorio `db/`.
4.  **Cadena Conversacional**: El componente [`src/chatbot.py`](src/chatbot.py) configura una cadena de recuperación conversacional (`ConversationalRetrievalChain`). Esta cadena integra el LLM de Google Gemini, el sistema de recuperación de ChromaDB y una memoria para mantener el historial de la conversación.
5.  **Interacción con el Usuario**: Finalmente, [`main.py`](main.py) inicia un bucle interactivo en la consola, permitiendo al usuario chatear con el bot en tiempo real.

## Instalación

Sigue estos pasos para configurar y ejecutar el proyecto en tu entorno local:

1.  **Clonar el Repositorio** (si aplica)
    ```bash
    git clone https://github.com/tu-usuario/tu-repositorio.git
    cd tu-repositorio
    ```

2.  **Crear un Entorno Virtual** (recomendado)
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate
    ```

3.  **Instalar Dependencias**
    El proyecto utiliza las dependencias listadas en el archivo [`requirements.txt`](requirements.txt). Instálalas con el siguiente comando:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configurar la API Key**
    Necesitas una API key de Google para usar el modelo Gemini.
    *   Crea un archivo llamado `.env` en la raíz del proyecto.
    *   Añade tu API key al archivo de la siguiente manera:
        ```
        GOOGLE_API_KEY="TU_API_KEY_AQUI"
        ```

## Uso

Para interactuar con el chatbot, sigue estas instrucciones:

1.  **Añadir Documentos**: Coloca todos los documentos que desees usar como base de conocimiento (archivos `.pdf`, `.docx`, `.txt`) dentro de la carpeta `data/`.

2.  **Ejecutar el Chatbot**: Inicia el script principal desde la terminal:
    ```bash
    python main.py
    ```

3.  **Interactuar**: Una vez que el script se esté ejecutando, verás un mensaje de bienvenida. Escribe tus preguntas en la consola y presiona Enter para recibir una respuesta del chatbot. Para salir, escribe "salir".
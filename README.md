# Building an Advanced Auto Retrieval-Augmented Generation (RAG) with OpenAI

## Project Overview

AutoRAG (Retrieval-Augmented Generation) Assistant is an advanced AI project designed to enhance the capabilities of a conversational assistant by integrating retrieval-based techniques. This project leverages OpenAI's language models, a vector database for embedding storage, and various document readers to provide precise and contextually relevant responses to user queries.

## Relevant Links

- [Web Application](https://advance-auto-RAG-assistant.streamlit.app/) -- [Do this before Use](#instructions)
- [GitHub](https://github.com/sahiltambe/Advanced-Retrieval-Augmented-Generation-RAG-with-OpenAI/)
- [LinkedIn](https://www.linkedin.com/in/sahiltambe13//)

## Objectives

- To create an assistant that combines generative AI with a retrieval-based approach.
- To store and manage documents in a vector database for efficient information retrieval.
- To enable the assistant to search both its knowledge base and the web for the most relevant answers.
- To ensure a smooth and interactive conversational experience for users.

## Features

- **Document Management**: Upload and manage documents in the knowledge base.
- **Vector Database Integration**: Use PgVector for storing and retrieving document embeddings.
- **Web Search Capability**: Search the web using DuckDuckGo for additional information.
- **Conversational AI**: Utilize OpenAI's language models to generate responses.
- **Chat History**: Maintain and reference chat history for contextually aware interactions.

## Implementation

### Dependencies

- `Python`
- `OpenAI API Key`
- `Your PDF Document`
- `phi` (Custom AI framework)
- `psycopg` (PostgreSQL adapter for Python if any)

## Instructions

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/sahiltambe/Advanced-Retrieval-Augmented-Generation-RAG-with-OpenAI.git
    ```

2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Set Up Environment and OpenAI API Key Variables**:
    - Obtain your OpenAI API key from [OpenAI](https://www.openai.com).
    - Create a `.env` file in the root directory and add your API key:
      ```env
      OPENAI_MODEL_NAME = "OPENAI_MODEL_NAME"
      OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
      ```

4. **Setup your Docker**:   
    
    - Open Your Docker Container and Connect it with a `db_url` variable inside code `main.py` for connecting to the PostgreSQL database like a did (e.g `"db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"`)
    
    OR

    - Create and connect it with your database

5. **Run the Streamlit App**:
    ```bash
    streamlit run main.py
    ```

6. **Interact with the Assistant**:

    - Add documents to the knowledge base.
    - Choose a PDF file.
    - Query the assistant.


## Contributing
If you would like to contribute to this project, please fork the repository and submit a pull request. Ensure that your code adheres to the project's coding standards and includes appropriate tests.

## Contact
For any questions or support, please contact [sahiltambe1996@gmail.com](mailto:sahiltambe1996@gmail.com).

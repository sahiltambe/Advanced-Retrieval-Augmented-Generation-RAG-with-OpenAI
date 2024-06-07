import streamlit as st
import nest_asyncio
import logging
import os
from typing import List

from phi.assistant import Assistant
from phi.document import Document
from phi.document.reader.pdf import PDFReader
from phi.document.reader.website import WebsiteReader
from phi.llm.openai import OpenAIChat
from phi.knowledge import AssistantKnowledge
from phi.tools.duckduckgo import DuckDuckGo
from phi.embedder.openai import OpenAIEmbedder
from phi.vectordb.pgvector import PgVector2
from phi.storage.assistant.postgres import PgAssistantStorage
import psycopg

# Set up environment variables
os.environ['OPENAI_APT_KEY'] = 'OPENAI_APT_KEY'
api_key = os.environ['OPENAI_APT_KEY']

# Database URL for connecting to PostgreSQL
db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai" # Insert your Docker container link for saving the vector database
logger = logging.getLogger(__name__)

# Setup Assistant
def setup_assistant(llm: str) -> Assistant:
    return Assistant(
        name="auto_rag_assistant",
        llm=llm,
        storage=PgAssistantStorage(table_name="auto_rag_assistant_openai", db_url=db_url), # add db_url=db_url as second parameter
        knowledge_base=AssistantKnowledge(
            vector_db=PgVector2(
                db_url=db_url,
                collection="auto_rag_documents_openai",
                embedder=OpenAIEmbedder(model="text-embedding-3-small", api_key=api_key, dimensions=1536),
            ),
            num_documents=3,
        ),
        description="Your job as a helpful assistant named 'AutoRAG' is to aid the user as much as you can.",
        instructions=[
            "Always use the `search_knowledge_base` tool to search your knowledge base for relevant information when responding to a user query.",

            "Use the `duckduckgo_search` tool to search the internet if you are unable to find relevant material in your knowledge base.""Use the `get_chat_history} tool if you need to refer to the chat history."

            "Ask clarifying questions to obtain further information if the user's inquiry is unclear."

            "Read the content carefully and respond to the user in a clear and succinct manner."

            "Sayings like 'based on my expertise' or 'depending on the information' should be avoided.",
        ],
        show_tool_calls=True,
        search_knowledge=True,
        read_chat_history=True,
        tools=[DuckDuckGo()],
        markdown=True,
        add_chat_history_to_messages=True,
        add_datetime_to_instructions=True,
        debug_mode=True,
    )

# Add Document to Knowledge Base
def add_document_to_kb(assistant: Assistant, file_path: str, file_type: str = "pdf"):
    if file_type == "pdf":
        reader = PDFReader()
    else:
        raise ValueError("Unsupported file type")
    documents: List[Document] = reader.read(file_path)
    if documents:
        assistant.knowledge_base.load_documents(documents, upsert=True)
        logger.info(f"Document '{file_path}' added to the knowledge base.")
    else:
        logger.error("Could not read document")

# Run Query
def query_assistant(assistant: Assistant, question: str):
    response = ""
    for delta in assistant.run(question):
        response += delta  # type: ignore
    return response

# Streamlit app
def main():
    st.title("Your Advance Auto RAG Assistant")
    
    # Initialize Assistant
    nest_asyncio.apply()
    llm_model = os.getenv("OPENAI_MODEL_NAME", "gpt-4")
    llm = OpenAIChat(model=llm_model, api_key=api_key)
    assistant = setup_assistant(llm)
    
    # Sidebar for adding documents
    st.sidebar.title("Add Documents to Knowledge Base")
    uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        add_document_to_kb(assistant, uploaded_file.name, file_type="pdf")
        st.sidebar.success("Document added to the knowledge base")
    
    # Query section
    st.header("Ask a Question")
    user_query = st.text_input("Enter your question:")
    if st.button("Submit"):
        if user_query:
            response = query_assistant(assistant, user_query)
            st.write("**Response:**")
            st.write(response)
        else:
            st.write("Please enter a question.")

if __name__ == "__main__":
    main()
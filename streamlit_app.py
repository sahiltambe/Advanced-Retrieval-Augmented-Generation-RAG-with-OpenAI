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
# import psycopg # add psycopg in requirements.txt

# Set up environment variables
os.environ['OPENAI_APT_KEY'] = st.secrets["OPENAI_APT_KEY"]

# Database URL for connecting to PostgreSQL
# db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"
logger = logging.getLogger(__name__)

# Setup Assistant
def setup_assistant(llm: str) -> Assistant:
    return Assistant(
        name="auto_rag_assistant",
        llm=llm,
        storage=PgAssistantStorage(table_name="auto_rag_assistant_openai", db_url=db_url),
        knowledge_base=AssistantKnowledge(
            vector_db=PgVector2(
                db_url=db_url,
                collection="auto_rag_documents_openai",
                embedder=OpenAIEmbedder(model="text-embedding-3-small", api_key='OPENAI_APT_KEY', dimensions=1536),
            ),
            num_documents=3,
        ),
        description="You are a helpful Assistant called 'AutoRAG' and your goal is to assist the user in the best way possible.",
        instructions=[
            "Given a user query, first ALWAYS search your knowledge base using the `search_knowledge_base` tool to see if you have relevant information.",
            "If you don't find relevant information in your knowledge base, use the `duckduckgo_search` tool to search the internet.",
            "If you need to reference the chat history, use the `get_chat_history` tool.",
            "If the user's question is unclear, ask clarifying questions to get more information.",
            "Carefully read the information you have gathered and provide a clear and concise answer to the user.",
            "Do not use phrases like 'based on my knowledge' or 'depending on the information'.",
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
    llm = OpenAIChat(model=llm_model, api_key='OPENAI_APT_KEY')
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

from typing import List
import nest_asyncio    # Ensures that asynchronous code runs smoothly in environments that may already have an event loop running.

import logging         # Standard Python library used for tracking events with different levels of severity.

import os              # Used for tasks like reading environment variables or manipulating file paths.

from phi.assistant import Assistant   # Handles conversational interactions, incorporating responses from various sources and knowledge bases.
from phi.document import Document     # Creates, reads, or manipulates documents for the assistant.
from phi.document.reader.pdf import PDFReader   # Reads PDF documents.
from phi.document.reader.website import WebsiteReader # Reads content from URLs.
from phi.llm.openai import OpenAIChat  # Integrates OpenAI's language models for chat functionality.

from phi.knowledge import AssistantKnowledge  # Manages the knowledge base for the assistant.
from phi.tools.duckduckgo import DuckDuckGo   # Enables web search functionality.
from phi.embedder.openai import OpenAIEmbedder # Generates embeddings for documents.
from phi.vectordb.pgvector import PgVector2    # Stores embeddings in a vector database.
from phi.storage.assistant.postgres import PgAssistantStorage  # Stores chat history and other assistant data.
import psycopg  # PostgreSQL adapter for Python.

# Set OpenAI API key as an environment variable
os.environ['OPENAI_APT_KEY'] = 'OPENAI_APT_KEY'

# Database URL for connecting to the PostgreSQL database
db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"  # Docker container for saving the vector database
logger = logging.getLogger(__name__)

# Function to setup the assistant
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

# Function to add a document to the knowledge base
def add_document_to_kb(assistant: Assistant, file_path: str, file_type: str = "pdf"):
    if file_type == "pdf":
        reader = PDFReader()  # Initialize PDF reader
    else:
        raise ValueError("Unsupported file type")  # Handle unsupported file types
    documents: List[Document] = reader.read(file_path)  # Read the document
    if documents:
        assistant.knowledge_base.load_documents(documents, upsert=True)  # Load documents into the knowledge base
        logger.info(f"Document '{file_path}' added to the knowledge base.")
    else:
        logger.error("Could not read document")

# Function to query the assistant
def query_assistant(assistant: Assistant, question: str):
    response = ""
    for delta in assistant.run(question):
        response += delta  # Append response fragments
    return response

if __name__ == "__main__":
    nest_asyncio.apply()  # Apply nest_asyncio to ensure smooth async operation
    llm_model = os.getenv("OPENAI_MODEL_NAME", "gpt-4")  # Get the LLM model name from environment variable
    llm = OpenAIChat(model=llm_model, api_key='OPENAI_APT_KEY')  # Initialize OpenAI chat with model and API key
    assistant = setup_assistant(llm)  # Setup the assistant
    sample_pdf_path = "Parameter-Efficient Transfer Learning for NLP.pdf"  # Path to sample PDF
    add_document_to_kb(assistant, sample_pdf_path, file_type="pdf")  # Add document to knowledge base
    query = "What is Adapter tuning for NLP?"  # Sample query
    # query = "Your Specific Question?" # Your Sample query
    response = query_assistant(assistant, query)  # Query the assistant
    print("Query:", query)
    print("Response:", response)

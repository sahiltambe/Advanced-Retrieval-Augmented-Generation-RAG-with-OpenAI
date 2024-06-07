# Environment Variables
import nest_asyncio
import logging
import os

nest_asyncio.apply()
os.environ['GOOGLE_API_KEY'] = 'GOOGLE_API_KEY'

# Initializing the Assistant

from phi.assistant import Assistant
from phi.llm.google import GooglePalmChat  # Assuming a hypothetical GooglePalmChat integration
from phi.knowledge import AssistantKnowledge
from phi.embedder.google import GoogleEmbedder  # Assuming a hypothetical GoogleEmbedder integration
from phi.vectordb.pgvector import PgVector2
from phi.storage.assistant.postgres import PgAssistantStorage
import psycopg

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"
logger = logging.getLogger(__name__)

def setup_assistant(llm: str) -> Assistant:
    return Assistant(
        name="auto_rag_assistant",
        llm=llm,
        storage=PgAssistantStorage(table_name="auto_rag_assistant_google", db_url=db_url),
        knowledge_base=AssistantKnowledge(
            vector_db=PgVector2(
                db_url=db_url,
                collection="auto_rag_documents_google",
                embedder=GoogleEmbedder(model="palm-embedding-small", api_key=os.getenv('GOOGLE_API_KEY'), dimensions=1536),
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
        tools=[DuckDuckGo()], # type: ignore
        markdown=True,
        add_chat_history_to_messages=True,
        add_datetime_to_instructions=True,
        debug_mode=True,
    )

llm_model = "palm"  # Assuming 'palm' is the identifier for Google Palm model
llm = GooglePalmChat(model=llm_model, api_key=os.getenv('GOOGLE_API_KEY'))
assistant = setup_assistant(llm)

# Adding Documents to the Knowledge Base

from typing import List
from phi.document import Document
from phi.document.reader.pdf import PDFReader

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

# Querying the Assistant

def query_assistant(assistant: Assistant, question: str):
    response = ""
    for delta in assistant.run(question):
        response += delta  # type: ignore
    return response

sample_pdf_path = "Parameter-Efficient Transfer Learning for NLP.pdf"
add_document_to_kb(assistant, sample_pdf_path, file_type="pdf")
query = "Your Specific Question?"
response = query_assistant(assistant, query)
print("Query:", query)
print("Response:", response)

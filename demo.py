import streamlit as st
import nest_asyncio
import logging
import os

# Set up environment variables
# os.environ['OPENAI_APT_KEY'] = st.secrets["OPENAI_APT_KEY"]
os.environ['OPENAI_APT_KEY'] = 'OPENAI_APT_KEY'

# Initialize Streamlit
nest_asyncio.apply()

# UI Layout
def main():
    # Link to GitHub repository
    st.markdown("[GitHub Repository](https://github.com/sahiltambe/Advanced-Retrieval-Augmented-Generation-RAG-with-OpenAI)", unsafe_allow_html=True)

    
    st.markdown("""
    **Note: This is a demonstration of application. To use this application effectively for your specific use case, 
    please ensure you have completed the necessary pre-requisites as outlined in the <a href="https://github.com/sahiltambe/Advanced-Retrieval-Augmented-Generation-RAG-with-OpenAI?tab=readme-ov-file#instructions" target="_blank">Instructions</a>.** """,unsafe_allow_html=True)

    st.title("Your Advance Auto RAG Assistant")
    
    # Sidebar for adding documents
    st.sidebar.title("Add Documents to Knowledge Base")
    uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.sidebar.success("Document added to the knowledge base")
    
    # Main section for asking a question
    st.header("Ask a Question")
    user_query = st.text_input("Enter your question:")
    
    if st.button("Submit"):
        if user_query:
            st.info("Processing your query...")
            response = "This is a sample response. Your query will be processed in a real deployment."
            st.write("**Response:**")
            st.write(response)
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main()

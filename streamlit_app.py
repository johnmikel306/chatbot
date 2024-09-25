import os
import streamlit as st
from langchain import hub
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import NotionDirectoryLoader, UnstructuredFileIOLoader, GoogleDriveLoader
# from langchain_google_community import GoogleDriveLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import NotionDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from notion_client import Client
from dotenv import load_dotenv
from pprint import pprint

# Load environment variables from .env file
load_dotenv()

# Set environment variables for Google API credentials
SCOPES = ['https://www.googleapis.com/auth/drive']
os.environ['GOOGLE_ACCOUNT_FILE'] = "credentials.json"
os.environ['GOOGLE_API_KEY'] = "AIzaSyBO7d_cbzrxVDIxKj9MvG8OhedqC7Qt-L8"

# Function to initialize Notion client
def initialize_notion_client():
    """Initialize and return the Notion client using the token from environment variables."""
    return Client(auth=os.getenv("NOTION_TOKEN"))

# Function to load Notion documents with caching
@st.cache_data(show_spinner=True, max_entries=10)
def load_notion_documents(_notion_client):
    """Load documents from the Notion database and cache the result."""
    loader = NotionDirectoryLoader("Notion_DB")
    return loader.load()

# Function to load Google Drive documents with caching
@st.cache_data(show_spinner=True, max_entries=10)
def load_google_drive_documents():
    """Load documents from Google Drive and cache the result."""
    loader = GoogleDriveLoader(
        folder_id="0ANiSnGo3Uz8VUk9PVA",  # Replace with your folder ID
        credentials_path="credentials.json",
        token_path="token.json",
        file_types=["document", "sheet"],
        file_loader_cls=UnstructuredFileIOLoader,
        file_loader_kwargs={"mode": "elements"},
        recursive=False,
    )
    return loader.load()

# Function to combine and split documents
def split_documents(all_docs):
    """Split combined documents into chunks using a text splitter."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    splits = text_splitter.split_documents(all_docs)
    return splits

# Function to create vector store with caching
@st.cache_resource(show_spinner=True)
def create_vector_store(_splits):
    """Create and return the FAISS vector store using text chunks."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.from_documents(_splits, embeddings)

# Function to initialize and return the RAG chain
def create_rag_chain(db):
    """Create and return the Retrieval-Augmented Generation (RAG) chain."""
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    retriever = db.as_retriever(search_type='similarity', search_kwargs={"k": 4})
    prompt = hub.pull("rlm/rag-prompt")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# Main function for the Streamlit app
def main():
    """Main function to orchestrate the Streamlit app."""
    st.title("MIVA Success Advisor's Assistant")
    
    # Initialize Notion client
    notion_client = initialize_notion_client()
    
    # Load documents
    with st.spinner("Loading documents from Notion..."):
        notion_docs = load_notion_documents(notion_client)
    
    with st.spinner("Loading documents from Google Drive..."):
        drive_docs = load_google_drive_documents()
    
    # Combine and split documents
    all_docs = notion_docs + drive_docs
    splits = split_documents(all_docs)

    # Create vector store
    db = create_vector_store(splits)
    
    # Create RAG chain
    rag_chain = create_rag_chain(db)
    
    # Render the chat interface
    render_chat_interface(rag_chain)

# Function to handle the chat interface
def render_chat_interface(rag_chain):
    """Render the chat interface with user input and responses."""
    chat_history = st.session_state.get('chat_history', [])
    
    # Display chat history
    for message in chat_history:
        if message['sender'] == 'user':
            st.chat_message("user").write(message['content'])
        else:
            st.chat_message("ai").write(message['content'])

    # Handle user input
    user_question = st.chat_input(placeholder="Ask me anything!")
    if user_question:
        st.chat_message("user").write(user_question)
        response = ""
        for chunk in rag_chain.stream(user_question):
            response += chunk
        st.chat_message("ai").write(response)

        # Add the new chat to the history
        chat_history.append({'sender': 'user', 'content': user_question})
        chat_history.append({'sender': 'ai', 'content': response})

        # Save the chat history to session state
        st.session_state['chat_history'] = chat_history

# Run the main function when the script is executed
if __name__ == "__main__":
    main()

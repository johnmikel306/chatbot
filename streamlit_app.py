import os
import re
import json
import streamlit as st
from langchain import hub
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import UnstructuredFileIOLoader, GoogleDriveLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory

# Access secrets from Streamlit Secrets Manager
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
TOKEN_PATH = st.secrets['TOKEN_PATH']

# Parse the nested secrets
google_account_info = {
    "installed": {
        "client_id": st.secrets["GOOGLE_ACCOUNT_FILE"]["installed"]["client_id"],
        "project_id": st.secrets["GOOGLE_ACCOUNT_FILE"]["installed"]["project_id"],
        "auth_uri": st.secrets["GOOGLE_ACCOUNT_FILE"]["installed"]["auth_uri"],
        "token_uri": st.secrets["GOOGLE_ACCOUNT_FILE"]["installed"]["token_uri"],
        "auth_provider_x509_cert_url": st.secrets["GOOGLE_ACCOUNT_FILE"]["installed"]["auth_provider_x509_cert_url"],
        "client_secret": st.secrets["GOOGLE_ACCOUNT_FILE"]["installed"]["client_secret"],
        "redirect_uris": json.loads(st.secrets["GOOGLE_ACCOUNT_FILE"]["installed"]["redirect_uris"])
    }
}

google_token_info = {
    "token": st.secrets["GOOGLE_TOKEN_FILE"]["token"],
    "refresh_token": st.secrets["GOOGLE_TOKEN_FILE"]["refresh_token"],
    "token_uri": st.secrets["GOOGLE_TOKEN_FILE"]["token_uri"],
    "client_id": st.secrets["GOOGLE_TOKEN_FILE"]["client_id"],
    "client_secret": st.secrets["GOOGLE_TOKEN_FILE"]["client_secret"],
    "scopes": json.loads(st.secrets["GOOGLE_TOKEN_FILE"]["scopes"]),
    "universe_domain": st.secrets["GOOGLE_TOKEN_FILE"]["universe_domain"],
    "account": st.secrets["GOOGLE_TOKEN_FILE"]["account"],
    "expiry": st.secrets["GOOGLE_TOKEN_FILE"]["expiry"]
}

# Write them to temporary files if required by libraries
credentials_path = "/tmp/credentials.json"
token_path = TOKEN_PATH
with open(credentials_path, 'w') as f:
    json.dump(google_account_info, f)
with open(token_path, 'w') as f:
    json.dump(google_token_info, f)

# Set environment variables for Google API credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path

# Initialize persistent memory
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )


# Function to load Google Drive documents with caching
@st.cache_data(show_spinner=True, max_entries=10)
def load_google_drive_documents():
    """Load documents from Google Drive and cache the result."""
    loader = GoogleDriveLoader(
        folder_id= st.secrets['GOOGLE_DRIVE_FOLDER_ID'],
        credentials_path=credentials_path,
        token_path=token_path,
        # file_types=["document", "sheet"],
        file_loader_cls=UnstructuredFileIOLoader,
        file_loader_kwargs={"mode": "elements"},
        recursive=False,
    )
    all_docs = loader.load()
    return all_docs

# Function to combine and split documents
def split_documents(all_docs):
    """Split combined documents into chunks using a text splitter."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=150,
        length_function=len,
        is_separator_regex=False,
    )
    splits = text_splitter.split_documents(all_docs)
    return splits

# Function to create vector store with caching
@st.cache_resource(show_spinner=True)
def create_vector_store(_splits):
    """Create and return the FAISS vector store using text chunks."""
    embeddings = GoogleGenerativeAIEmbeddings(api_key=GOOGLE_API_KEY, model="models/embedding-001")
    return FAISS.from_documents(_splits, embeddings)

# Function to initialize and return the RAG chain
def create_rag_chain(db):
    """Create and return the Retrieval-Augmented Generation (RAG) chain."""
    retriever = db.as_retriever(search_type='similarity', search_kwargs={"k": 6})
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", streaming=True, temperature = 0.7)

    prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant for MIVA Success Advisors. Use the following pieces of context to answer the human's question. Include all relevant information, including sensitive data such as phone numbers and email addresses. Do not redact or omit any information. If you don't know the answer, just say that you don't know. Always maintain context from the chat history provided.\n\nContext: {context}"),
    ("human", "Chat History:\n{chat_history}\n\nHuman: {question}"),
    ("ai", "Assistant: ")
]) 

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def format_chat_history(chat_history):
        formatted_history = []
        for message in chat_history:
            if message.type == 'human':
                formatted_history.append(f"Human: {message.content}")
            elif message.type == 'ai':
                formatted_history.append(f"Assistant: {message.content}")
        return "\n".join(formatted_history)

    rag_chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.get_relevant_documents(x["question"])),
            chat_history=lambda x: format_chat_history(x["chat_history"])
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# Main function for the Streamlit app
def main():
    """Main function to orchestrate the Streamlit app."""
    st.title("Knowledge Base Retreiver")
    
    with st.spinner("Loading documents from Google Drive..."):
        all_docs = load_google_drive_documents()
    
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
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    if user_question := st.chat_input("Ask me anything!"):
        st.session_state.messages.append({"role": "human", "content": user_question})
        st.chat_message("human").markdown(user_question)

        with st.chat_message("ai"):
            response_placeholder = st.empty()
            full_response = ""
            for chunk in rag_chain.stream({
                "question": user_question,
                "chat_history": st.session_state.memory.chat_memory.messages
            }):
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")
            response_placeholder.markdown(full_response)
        
        st.session_state.messages.append({"role": "ai", "content": full_response})
        st.session_state.memory.chat_memory.add_user_message(user_question)
        st.session_state.memory.chat_memory.add_ai_message(full_response)


# Run the main function when the script is executed
if __name__ == "__main__":
    main()

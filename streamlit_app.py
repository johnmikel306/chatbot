import os
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

# Access secrets from Streamlit Secrets Manager
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
GOOGLE_DRIVE_FOLDER_ID = st.secrets['GOOGLE_DRIVE_FOLDER_ID']
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

# Function to load Google Drive documents with caching
@st.cache_data(show_spinner=True, max_entries=10)
def load_google_drive_documents():
    """Load documents from Google Drive and cache the result."""
    loader = GoogleDriveLoader(
        folder_id=GOOGLE_DRIVE_FOLDER_ID,
        credentials_path=credentials_path,
        token_path=token_path,
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
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.from_documents(_splits, embeddings)

# Function to initialize and return the RAG chain
def create_rag_chain(db):
    """Create and return the Retrieval-Augmented Generation (RAG) chain."""
    retriever = db.as_retriever(search_type='similarity', search_kwargs={"k": 6})
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    prompt = hub.pull("rlm/rag-prompt")
    # prompt = ChatPromptTemplate.from_messages([
    # ("human", """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know as it is not included in the Google Drive you're provided.
    # For phone numbers, always make sure to add "+234" infront of the number if there is no "0" as the first number, else replace the starting "0" with "+234".
    # For amounts in Naira, add the Naira sign infront of the figure. Do the same for amounts in dollars. 
    # When asked questions whose answers have urls included, make sure to provide the url embedded in the response as well.
    # For emails responses, draft it as a success/programs advisor with 10 years of experience in academic advising.
    # Make it very conversational and human-like but make sure to only provide relevant information. Make sure to provide empathy where necessary.
    # Question: {question} 
    # Context: {context} 
    # Answer:"""),
    # ])

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
    
    with st.spinner("Loading documents from Google Drive..."):
        drive_docs = load_google_drive_documents()
    
    # Combine and split documents
    # all_docs = notion_docs + drive_docs
    splits = split_documents(drive_docs)

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

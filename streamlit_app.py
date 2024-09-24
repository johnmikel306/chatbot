import streamlit as st
import os
import json
import io
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from langchain import hub
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from notion_client import Client
from langchain_core.documents import Document
from langchain_community.document_loaders import NotionDirectoryLoader

# Streamlit page config
st.set_page_config(layout="wide", page_title="MIVA Success Advisor's Assistant")

# Initialize session state
if 'db' not in st.session_state:
    st.session_state['db'] = None
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Google Drive Authentication and Data Loading
def authenticate_google_drive():
    creds = None
    if 'token' in st.session_state:
        creds = Credentials.from_authorized_user_info(json.loads(st.session_state['token']))
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = Flow.from_client_config(
                json.loads(st.secrets["GOOGLE_CREDENTIALS"]),
                scopes=['https://www.googleapis.com/auth/drive.readonly']
            )
            # Use run_console for environments like Streamlit Cloud
            creds = flow.run_console()

        # Save the credentials for the next run
        st.session_state['token'] = creds.to_json()
    
    return creds

@st.cache_resource
def load_google_drive_data():
    creds = authenticate_google_drive()
    if not creds:
        st.error("Failed to authenticate with Google Drive.")
        return []

    folder_id = st.secrets["GOOGLE_DRIVE_FOLDER_ID"]
    if not folder_id:
        st.error("Google Drive folder ID is not set. Please check your Streamlit secrets.")
        return []

    try:
        drive_service = build('drive', 'v3', credentials=creds)
        results = drive_service.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            fields="files(id, name, mimeType)"
        ).execute()
        files = results.get('files', [])

        documents = []
        for file in files:
            if file['mimeType'] in ['application/vnd.google-apps.document', 'application/vnd.google-apps.spreadsheet']:
                content = download_file_content(drive_service, file['id'], file['mimeType'])
                documents.append({
                    'name': file['name'],
                    'content': content
                })

        return documents
    except Exception as e:
        st.error(f"An error occurred while accessing Google Drive: {str(e)}")
        return []

def download_file_content(drive_service, file_id, mime_type):
    try:
        if mime_type == 'application/vnd.google-apps.document':
            request = drive_service.files().export_media(fileId=file_id, mimeType='text/plain')
        elif mime_type == 'application/vnd.google-apps.spreadsheet':
            request = drive_service.files().export_media(fileId=file_id, mimeType='text/csv')
        else:
            return ""

        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()

        return fh.getvalue().decode('utf-8')
    except Exception as e:
        st.error(f"Error downloading file content: {str(e)}")
        return ""

# Notion Data Loading
@st.cache_resource
def load_notion_data():
    try:
        notion_token = st.secrets["NOTION_TOKEN"]
        if not notion_token:
            st.error("Notion token is not set. Please check your Streamlit secrets.")
            return []
        
        client = Client(auth=notion_token)
        loader = NotionDirectoryLoader("Notion_DB")  # Assumes a folder path named "Notion_DB"
        return loader.load()
    except Exception as e:
        st.error(f"Error loading Notion data: {str(e)}")
        return []

# Vector Store Creation
@st.cache_resource
def create_vector_store(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    splits = text_splitter.split_documents(documents)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.from_documents(splits, embeddings)

# RAG Chain Initialization
def initialize_rag_chain():
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    retriever = st.session_state['db'].as_retriever(search_type='similarity', search_kwargs={"k": 6})
    prompt = hub.pull("rlm/rag-prompt")
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

# Main Streamlit App
def main():
    st.title("MIVA Success Advisor's Assistant")

    # Load and process data if not already done
    if st.session_state['db'] is None:
        with st.spinner("Loading and processing data..."):
            try:
                notion_docs = load_notion_data()
                google_docs = load_google_drive_data()
                
                # Combine Notion and Google Drive documents
                all_docs = notion_docs + [
                    Document(page_content=doc['content'], metadata={"source": doc['name']})
                    for doc in google_docs
                ]
                
                if not all_docs:
                    st.warning("No documents were loaded. Please check your data sources and permissions.")
                    return

                st.session_state['db'] = create_vector_store(all_docs)
                st.success("Data loaded successfully!")
            except Exception as e:
                st.error(f"An error occurred while loading data: {str(e)}")
                return

    # Initialize RAG chain
    rag_chain = initialize_rag_chain()

    # Chat Interface
    st.subheader("Chat with the Assistant")
    
    # Render chat history
    for message in st.session_state['chat_history']:
        with st.chat_message(message['sender']):
            st.write(message['content'])

    # Chat input
    user_question = st.chat_input(placeholder="Ask me anything!")
    if user_question:
        st.session_state['chat_history'].append({'sender': 'user', 'content': user_question})
        with st.chat_message("user"):
            st.write(user_question)

        with st.chat_message("ai"):
            response_container = st.empty()
            response = ""
            for chunk in rag_chain.stream(user_question):
                response += chunk
                response_container.markdown(response + "▌")
            response_container.markdown(response)
        
        st.session_state['chat_history'].append({'sender': 'ai', 'content': response})

if __name__ == "__main__":
    main()

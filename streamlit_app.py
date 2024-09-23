import os
import streamlit as st
from langchain import hub
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import NotionDirectoryLoader, UnstructuredFileIOLoader
from langchain_google_community import GoogleDriveLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from notion_client import Client
from dotenv import load_dotenv
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import Flow

# Load environment variables
load_dotenv()

# Streamlit config
st.set_page_config(layout="wide", page_title="MIVA Success Advisor's Assistant")

# Initialize session state
if 'db' not in st.session_state:
    st.session_state['db'] = None
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

@st.cache_resource
def load_notion_data():
    client = Client(auth=st.secrets["NOTION_TOKEN"])
    loader = NotionDirectoryLoader("Notion_DB")
    return loader.load()

@st.cache_resource
def authenticate_google_drive():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json')
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = Flow.from_client_secrets_file(
                'credentials.json',
                scopes=['https://www.googleapis.com/auth/drive.readonly']
            )
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return creds

@st.cache_resource
def load_google_drive_data(_credentials):
    folder_id = st.secrets["GOOGLE_DRIVE_FOLDER_ID"]
    if not folder_id:
        st.error("Google Drive folder ID is not set. Please check your .env file or Streamlit secrets.")
        return []
    
    loader = GoogleDriveLoader(
        folder_id=folder_id,
        credentials=_credentials,
        file_types=["document", "sheet"],
        file_loader_cls=UnstructuredFileIOLoader,
        file_loader_kwargs={"mode": "elements"},
        recursive=False,
    )
    return loader.load()

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

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def initialize_rag_chain():
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    retriever = st.session_state['db'].as_retriever(search_type='similarity', search_kwargs={"k": 6})
    prompt = hub.pull("rlm/rag-prompt")
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

def main():
    st.title("MIVA Success Advisor's Assistant")

    # Load and process data if not already done
    if st.session_state['db'] is None:
        with st.spinner("Loading and processing data..."):
            try:
                notion_docs = load_notion_data()
                google_credentials = authenticate_google_drive()
                google_docs = load_google_drive_data(google_credentials)
                all_docs = notion_docs + google_docs
                st.session_state['db'] = create_vector_store(all_docs)
            except Exception as e:
                st.error(f"An error occurred while loading data: {str(e)}")
                return

    rag_chain = initialize_rag_chain()

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

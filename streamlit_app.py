import streamlit as st
from langchain import hub
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import NotionDirectoryLoader, UnstructuredFileIOLoader
from langchain_community.document_loaders import GoogleDriveLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from notion_client import Client
import os
import json

# Set up Streamlit page configuration
st.set_page_config(layout="wide", page_title="MIVA Success Advisor's Assistant")

# Initialize Notion client with token from secrets
client = Client(auth=st.secrets["NOTION_TOKEN"])

# Loading the Notion database
loader1 = NotionDirectoryLoader("Notion_DB")
doc1 = loader1.load()

# Loading Google Drive with credentials from Streamlit secrets
loader2 = GoogleDriveLoader(
    folder_id=st.secrets["GOOGLE_DRIVE_FOLDER_ID"],
    credentials_path=None,  # Not used since we are passing the credentials via st.secrets
    token_path=None,  # Not used in Streamlit Cloud
    credentials_info=json.loads(st.secrets["GOOGLE_CREDENTIALS"]),  # Use the credentials JSON stored in Streamlit secrets
    file_types=["document", "sheet"],
    file_loader_cls=UnstructuredFileIOLoader,
    file_loader_kwargs={"mode": "elements"},
    recursive=False,
)
doc2 = loader2.load()

# Combine documents from Notion and Google Drive
all_docs = doc1 + doc2

# Split the texts into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)
splits = text_splitter.split_documents(all_docs)

# Create the vector store with embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
db = FAISS.from_documents(splits, embeddings)

# Initialize the language model and retriever
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
retriever = db.as_retriever(search_type='similarity', search_kwargs={"k": 6})

# Pull the RAG prompt template from LangChain hub
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Set up the RAG chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Streamlit app interface
st.title("MIVA Success Advisor's Assistant")

# Load chat history from session state (if available)
chat_history = st.session_state.get('chat_history', [])

# Render the chat history
for message in chat_history:
    if message['sender'] == 'user':
        st.chat_message("user").write(message['content'])
    else:
        st.chat_message("ai").write(message['content'])

# Chat input
user_question = st.chat_input(placeholder="Ask me anything!")
if user_question:
    st.chat_message("user").write(user_question)

    # Generate a response from the RAG chain
    response = ""
    for chunk in rag_chain.stream(user_question):
        response += chunk
    st.chat_message("ai").write(response)

    # Add the conversation to the chat history
    chat_history.append({'sender': 'user', 'content': user_question})
    chat_history.append({'sender': 'ai', 'content': response})

    # Save chat history to session state
    st.session_state['chat_history'] = chat_history

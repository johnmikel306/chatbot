import os
import streamlit as st
from notion_client import Client
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import NotionDirectoryLoader, UnstructuredFileIOLoader
from langchain_google_community import GoogleDriveLoader
from langchain_googledrive.document_loaders import GoogleDriveLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain import hub

# Retrieve environment variables
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
GOOGLE_ACCOUNT_FILE = st.secrets["GOOGLE_ACCOUNT_FILE"]
GOOGLE_DRIVE_FOLDER_ID = st.secrets["GOOGLE_DRIVE_FOLDER_ID"]
NOTION_TOKEN = st.secrets["NOTION_TOKEN"]

# Initialize Notion client
notion_client = Client(auth=NOTION_TOKEN)

@st.cache_data(show_spinner=False)
def load_notion_data():
    loader = NotionDirectoryLoader("Notion_DB")
    return loader.load()

@st.cache_data(show_spinner=False)
def load_google_drive_data():
    loader = GoogleDriveLoader(
        folder_id=GOOGLE_DRIVE_FOLDER_ID,
        credentials_path=GOOGLE_ACCOUNT_FILE,
        token_path="/tmp/token.json",
        file_types=["document", "sheet"],
        file_loader_cls=UnstructuredFileIOLoader,
        file_loader_kwargs={"mode": "elements"},
        recursive=False,
    )
    return loader.load()
    
def process_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(docs)

@st.cache_data(show_spinner=False)
def create_vector_store(splits):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.from_documents(splits, embeddings)

def create_rag_chain(vector_store):
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={"k": 6})
    prompt = hub.pull("rlm/rag-prompt")

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

def main():
    st.title("MIVA Success Advisor's Assistant")

    # Load data
    with st.spinner("Loading Notion data..."):
        notion_docs = load_notion_data()
    with st.spinner("Loading Google Drive data..."):
        drive_docs = load_google_drive_data()

    # Combine and process data
    all_docs = notion_docs + drive_docs
    with st.spinner("Processing documents..."):
        splits = process_documents(all_docs)
        vector_store = create_vector_store(splits)

    rag_chain = create_rag_chain(vector_store)

    # Chat Interface
    chat_history = st.session_state.get('chat_history', [])
    for message in chat_history:
        if message['sender'] == 'user':
            st.chat_message("user").write(message['content'])
        else:
            st.chat_message("ai").write(message['content'])

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

if __name__ == "__main__":
    main()


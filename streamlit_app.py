import os
import streamlit as st
from langchain_community.document_loaders import NotionDirectoryLoader, UnstructuredFileIOLoader
from langchain_google_community import GoogleDriveLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from notion_client import Client
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up Streamlit page configuration
st.set_page_config(page_title="MIVA Success Advisor's Assistant", layout="wide")

# Sidebar configuration
st.sidebar.title("Configuration")
api_key = st.sidebar.text_input("Enter your Google API Key:", type="password")

def load_notion_data():
    """Load data from Notion database."""
    client = Client(auth=os.getenv("NOTION_TOKEN"))
    loader = NotionDirectoryLoader("Notion_DB")
    return loader.load()

def load_google_drive_data():
    """Load data from Google Drive."""
    loader = GoogleDriveLoader(
        folder_id=os.getenv("GOOGLE_DRIVE_FOLDER_ID"),
        credentials_path=os.getenv("GOOGLE_ACCOUNT_FILE"),
        token_path="/workspaces/MyLLM-App/token.json",
        file_types=["document", "sheet"],
        file_loader_cls=UnstructuredFileIOLoader,
        file_loader_kwargs={"mode": "elements"},
        recursive=False,
    )
    return loader.load()

def get_text_chunks(docs):
    """Split documents into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(docs)

def build_vector_store(text_chunks, api_key):
    """Build and save the vector store."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def get_conversational_chain(api_key):
    """Create a conversational chain for question answering."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, api_key):
    """Process user question and return a response."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain(api_key)
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

def main():
    """Main function for the Streamlit app."""
    st.header("MIVA Success Advisor's Assistant")

    # Load documents and build vector store on submission
    if st.sidebar.button("Load & Process Data") and api_key:
        with st.spinner("Loading documents and processing..."):
            notion_data = load_notion_data()
            drive_data = load_google_drive_data()
            all_docs = notion_data + drive_data
            text_chunks = get_text_chunks(all_docs)
            build_vector_store(text_chunks, api_key)
            st.success("Documents processed successfully.")

    # User input and question handling
    user_question = st.text_input("Ask a Question from the Documents")
    if user_question and api_key:
        with st.spinner("Generating response..."):
            response = user_input(user_question, api_key)
            st.write("Reply:", response)

if __name__ == "__main__":
    main()

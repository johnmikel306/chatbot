import os
import streamlit as st
from langchain import hub
from langchain_google_genai import chatgooglegenerativeai
from langchain_core.prompts import chatprompttemplate
from langchain_community.document_loaders import notiondirectoryloader, unstructuredfileioloader
from langchain_google_community import googledriveloader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import googlegenerativeaiembeddings
from langchain_community.document_loaders import notiondirectoryloader
from langchain_text_splitters import recursivecharactertextsplitter
from langchain_core.output_parsers import stroutputparser
from langchain_core.runnables import runnablepassthrough
from langchain_community.vectorstores import faiss
from notion_client import Client
from dotenv import load_dotenv
from pprint import pprint

# Remove the load_dotenv call since we'll be using Streamlit secrets

# Configure environment variables using Streamlit secrets
google_account_file = st.secrets["google_account_file"]
os.environ['google_account_file'] = google_account_file

# initialize notion client with token from Streamlit secrets
notion_token = st.secrets["notion_token"]
client = Client(auth=notion_token)

# loading the notion database
loader1 = notiondirectoryloader("notion_db")
doc1 = loader1.load()

# loading the google drive
loader2 = googledriveloader(
    folder_id=st.secrets["google_drive_folder_id"],
    credentials_path="myllm-app/credentials.json",
    token_path="myllm-app/token.json",
    file_types=["document", "sheet"],
    file_loader_cls=unstructuredfileioloader,
    file_loader_kwargs={"mode": "elements"},
    recursive=False,
)
doc2 = loader2.load()

# combine documents from notion and google drive
all_docs = doc1 + doc2

# split the texts into chunks
text_splitter = recursivecharactertextsplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)
splits = text_splitter.split_documents(all_docs)

# create the vector store
embeddings = googlegenerativeaiembeddings(model="models/embedding-001")
db = faiss.from_documents(splits, embeddings)

llm = chatgooglegenerativeai(model="gemini-1.5-flash")

retriever = db.as_retriever(search_type='similarity', search_kwargs={"k": 6})

prompt = hub.pull("rlm/rag-prompt")
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": runnablepassthrough()}
    | prompt
    | llm
    | stroutputparser()
)

# streamlit app
st.title("MIVA Success Advisor's Assistant")
# st.logo("logo.jpg")  # Uncomment if you have a logo

# Load chat history from session state
chat_history = st.session_state.get('chat_history', [])

# Render the chat history
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

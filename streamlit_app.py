import os
import streamlit as st
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import UnstructuredFileIOLoader
from langchain_google_community import GoogleDriveLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS

# Retrieve environment variables
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# Function to load Google Drive documents with caching
@st.cache_data(show_spinner=True, max_entries=10)
def load_google_drive_documents():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = Flow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    loader = GoogleDriveLoader(
        folder_id=os.getenv("FOLDER_ID"),
        credentials=creds,
        file_types=["document", "sheet"],
        file_loader_cls=UnstructuredFileIOLoader,
        file_loader_kwargs={"mode": "elements"},
        recursive=False,
    )
    return loader.load()


# Function to combine and split documents
def split_documents(all_docs):
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
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.from_documents(_splits, embeddings)


# Function to initialize and return the RAG chain
def create_rag_chain(db):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                                 google_api_key=GOOGLE_API_KEY)
    retriever = db.as_retriever(search_type='similarity',
                                search_kwargs={"k": 4})
    prompt = ChatPromptTemplate.from_messages([
        ("human",
         """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know as it is not included in the Google Drive you're provided.
    For phone numbers, always make sure to add "+234" infront of the number if there is no "0" as the first number, else replace the starting "0" with "+234".
    For amounts in Naira, add the Naira sign infront of the figure. Do the same for amounts in dollars. 
    When asked questions whose answers have urls included, make sure to provide the url embedded in the response as well.
    For emails responses, draft it as a success/programs advisor with 10 years of experience in academic advising.
    Make it very conversational and human-like but make sure to only provide relevant information. Make sure to provide empathy where necessary.
    Question: {question} 
    Context: {context} 
    Answer:"""),
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = ({
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
                 | prompt
                 | llm
                 | StrOutputParser())
    return rag_chain


# Function to handle the chat interface
def render_chat_interface(rag_chain):
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

        chat_history.append({'sender': 'user', 'content': user_question})
        chat_history.append({'sender': 'ai', 'content': response})

        st.session_state['chat_history'] = chat_history


# Main function for the Streamlit app
def main():
    st.title("MIVA Success Advisor's Assistant")

    if not os.path.exists('token.json'):
        st.warning(
            "You need to authorize the application. Click the button below to start the authorization process."
        )
        if st.button("Authorize"):
            flow = Flow.from_client_secrets_file(
                'credentials.json',
                SCOPES,
                redirect_uri='urn:ietf:wg:oauth:2.0:oob')
            auth_url, _ = flow.authorization_url(prompt='consent')
            st.write(
                f"Please visit this URL to authorize the application: {auth_url}"
            )
            auth_code = st.text_input("Enter the authorization code:")
            if auth_code:
                flow.fetch_token(code=auth_code)
                creds = flow.credentials
                with open('token.json', 'w') as token:
                    token.write(creds.to_json())
                st.success(
                    "Authorization successful! You can now use the application."
                )
                st.experimental_rerun()
    else:
        with st.spinner("Loading documents from Google Drive..."):
            try:
                drive_docs = load_google_drive_documents()
            except Exception as e:
                st.error(f"Error loading documents: {str(e)}")
                return

        splits = split_documents(drive_docs)
        db = create_vector_store(splits)
        rag_chain = create_rag_chain(db)
        render_chat_interface(rag_chain)


# Run the main function when the script is executed
if __name__ == "__main__":
    main()

# Knowledge Base Retriever

Welcome to **Knowledge Base Retriever**, a powerful application designed to retrieve, process, and provide insights from your Google Drive documents using advanced AI technology. This project leverages Google's Gemini LLM and LangChain libraries for seamless Retrieval-Augmented Generation (RAG). 

---

## Features
- **Document Retrieval**: Automatically fetches and processes documents from Google Drive.
- **Vector-Based Search**: Creates a searchable database using FAISS and Google Generative AI embeddings.
- **Advanced Chat Interface**: Provides AI-powered, context-aware answers to user queries.
- **Persistent Memory**: Maintains conversation context across multiple queries using memory buffers.
- **Interactive Interface**: Built with Streamlit for an intuitive and user-friendly experience.

---

## Requirements

### Setup Secrets
Ensure your `secrets.toml` file in Streamlit contains:
```toml
[GOOGLE_ACCOUNT_FILE.installed]
client_id = "your-client-id"
project_id = "your-project-id"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_secret = "your-client-secret"
redirect_uris = ["http://localhost"]

[GOOGLE_TOKEN_FILE]
token = "your-token"
refresh_token = "your-refresh-token"
token_uri = "https://oauth2.googleapis.com/token"
client_id = "your-client-id"
client_secret = "your-client-secret"
scopes = '["your-required-scopes"]'
universe_domain = "your-domain"
account = "your-account"
expiry = "your-expiry-date"

GOOGLE_API_KEY = "your-google-api-key"
GOOGLE_DRIVE_FOLDER_ID = "your-folder-id"
TOKEN_PATH = "/tmp/token.json"
```

---

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Add your `secrets.toml` file to your Streamlit configuration.

4. Run the application:
   ```bash
   streamlit run app.py
   ```

---

## How It Works

1. **Document Retrieval**:
   - Fetches documents from Google Drive using the folder ID and credentials provided in the secrets file.
   - Supports text documents and spreadsheets.

2. **Text Splitting and Embeddings**:
   - Splits large documents into chunks for processing.
   - Creates embeddings for these chunks using Google Generative AI.

3. **Vector Store**:
   - Creates a FAISS vector store for efficient similarity search.

4. **Retrieval-Augmented Generation (RAG)**:
   - Combines retrieved document chunks with user queries to generate context-aware answers using Google's Gemini LLM.

5. **Interactive Chat**:
   - A Streamlit-powered chat interface allows users to interact with the AI.

---

## Usage

1. **Start the Application**:
   - Open the app in your browser after running the Streamlit command.

2. **Ask Questions**:
   - Enter your questions in the chat box, and the AI will provide relevant answers based on the uploaded documents.

3. **Persistent Memory**:
   - The app retains conversation context, making follow-up questions seamless.

---

## Dependencies

- [Streamlit](https://streamlit.io/)
- [LangChain](https://github.com/hwchase17/langchain)
- [Google Generative AI](https://cloud.google.com/generative-ai)
- [FAISS](https://github.com/facebookresearch/faiss)
- Python 3.8+

---

## Contributing
Feel free to contribute by creating issues or submitting pull requests. For major changes, please discuss with the repository owner first.

---

## License
This project is licensed under the MIT License.

---

Enjoy exploring your knowledge base with the power of AI! 🎉

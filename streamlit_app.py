import json
import streamlit as st
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build

@st.cache_resource
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
            creds = flow.run_local_server(port=0)
        
        # Save the credentials for the next run
        st.session_state['token'] = creds.to_json()
    
    return creds

def load_google_drive_data():
    creds = authenticate_google_drive()
    folder_id = st.secrets["GOOGLE_DRIVE_FOLDER_ID"]
    
    if not folder_id:
        st.error("Google Drive folder ID is not set. Please check your Streamlit secrets.")
        return []
    
    try:
        drive_service = build('drive', 'v3', credentials=creds)
        results = drive_service.files().list(
            q=f"'{folder_id}' in parents",
            fields="files(id, name, mimeType)"
        ).execute()
        files = results.get('files', [])
        
        # Here you would process these files as needed
        # For demonstration, we're just returning the file names
        return [file['name'] for file in files]
    except Exception as e:
        st.error(f"An error occurred while accessing Google Drive: {str(e)}")
        return []

# In your main function or Streamlit app:
def main():
    st.title("MIVA Success Advisor's Assistant")

    if st.button("Load Google Drive Data"):
        with st.spinner("Accessing Google Drive..."):
            files = load_google_drive_data()
            st.write("Files in the specified Google Drive folder:", files)

    # Rest of your Streamlit app code...

if __name__ == "__main__":
    main()

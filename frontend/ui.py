import requests
import streamlit as st

# --- Configuration ---
BACKEND_URL = "http://localhost:8000"

# App title
st.set_page_config(page_title="Study Buddy", page_icon = "🧠")
st.title("🧠 Study Buddy")

# Session State
if "processed" not in st.session_state:
    st.session_state.processed = False 
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Main App Logic ---
# 1. Document Upload in Sidebar
with st.sidebar:
    st.header("1. Upload Your Documents")
    uploaded_files= st.file_uploader(
        "Upload your PDF documents", type="pdf", accept_multiple_files=True
    )

    if st.button("Process Documents") and uploaded_files:
        with st.spinner("Processing documents..."):
            # Prepare files for the API request
            api_files= [("files", (file.name, file.getvalue(), file.type)) for file in uploaded_files]
            try:
                # send files to the backend
                response = requests.post(f"{BACKEND_URL}/upload/", files=api_files)
                if response.status_code == 200:
                    st.success("Documents processed successfully!")
                    st.session_state.processed = True
                else:
                    st.error(f"Error: {response.json().get('detail')}")
            except requests.exceptions.RequestException as e:
                st.error(f"Could not connect to backend: {e}")

# 2. Chat Interface
st.header("2. Ask Questions")

if st.session_state.processed: 
    # Display chat messages from histoy
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask a quesiton about your document(s)"):
        # Add user message to chat history
        st.session_state.messages.append({
            "role":"user",
            "content": prompt
        })
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get the assistant's response from the backend
        with st.chat_message("assistant"):
            with st.spinner("Thinking ... "):
                try:
                    response = requests.post(f"{BACKEND_URL}/ask/", data={"question": prompt})
                    if response.status_code == 200:
                        answer = response.json().get("answer")
                        st.markdown(answer)
                        # Add assistant response to chat histoy
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer
                        })
                    else:
                        st.error(f"Error: {response.json().get('detail')}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Could not get response from backend: {e}")
else:
    st.info("Please upload and process your documents in the sidebar to begin.")

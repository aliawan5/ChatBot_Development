import streamlit as st
import requests

FASTAPI_URL = "http://localhost:8000" 

st.title("Document Upload and Chatbot Interface")


st.header("Upload a Document")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    st.write(f"Uploaded file: {uploaded_file.name}")

    with st.spinner("Uploading document..."):
        files = {"file": uploaded_file.getvalue()}
        response = requests.post(f"{FASTAPI_URL}/upload", files=files)

        if response.status_code == 200:
            st.success("Document uploaded and processed successfully!")
        else:
            st.error(f"Error: {response.json()['detail']}")



st.header("Ask the Chatbot")
user_question = st.text_input("Ask a question about the document:")

if user_question:
    with st.spinner("Getting answer from chatbot..."):
        payload = {"question": user_question}
        response = requests.post(f"{FASTAPI_URL}/chat", json=payload)

        if response.status_code == 200:
            answer = response.json().get("answer")
            st.write(f"Answer: {answer}")
        else:
            st.error(f"Error: {response.json()['error']}")

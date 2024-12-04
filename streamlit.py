import streamlit as st
from graph import app
from langchain_core.messages import HumanMessage
import traceback
from PyPDF2 import PdfReader


st.set_page_config(page_title="Resume Skills Extractor", page_icon=":page_with_curl:")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

st.title("🤖 Resume Skills Extractor")

st.sidebar.title("Upload Resume")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")
if uploaded_file is None:
    st.sidebar.error("No resume uploaded. Please upload a PDF first.")
else:
    st.session_state.uploaded_file = uploaded_file
    st.sidebar.success("Resume uploaded successfully!")
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    st.session_state.resume_text = text
    st.sidebar.success("Resume uploaded successfully!")

    st.session_state.messages.append({
        "role": "system",
        "content": f"The user has uploaded the following resume text:\n{text}"
    })
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What would you like to know about the resume?"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            if "resume_text" in st.session_state:
                st.session_state.messages.append({
                    "role": "system",
                    "content": f"The resume text is:\n{st.session_state.resume_text}"
                })

            result = app.invoke({"messages": st.session_state.messages})

            if isinstance(result, dict) and 'messages' in result:
                assistant_response = result['messages'][-1]
                full_response = assistant_response.content if hasattr(assistant_response, 'content') else str(assistant_response)
            else:
                full_response = str(result)

            message_placeholder.markdown(full_response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
            message_placeholder.markdown(f"Error: {e}")

    st.session_state.messages.append({"role": "assistant", "content": full_response})


if st.button("Clear Chat"):
    st.session_state.messages = []
    st.session_state.uploaded_file = None
    st.experimental_rerun()
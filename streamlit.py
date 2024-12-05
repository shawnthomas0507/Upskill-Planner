import streamlit as st
from graph import app
from PyPDF2 import PdfReader

st.set_page_config(page_title="Upskill Planner", page_icon=":page_with_curl:")

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
    
    st.session_state.messages.append({
        "role": "system",
        "content": f"Resume text: {text}"
    })

for message in st.session_state.messages:
    if message["role"] != "system":  # Only display non-system messages
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to know about the resume?"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
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

# Clear chat button
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.session_state.uploaded_file = None
    st.experimental_rerun()
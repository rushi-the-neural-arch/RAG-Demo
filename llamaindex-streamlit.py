import streamlit as st
import os, openai
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, ServiceContext, SimpleDirectoryReader

from dotenv import load_dotenv
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Knowledge Extraction from your documents", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
# openai.api_key = st.secrets.openai_key
st.title("A demo on knowledge extraction from various data sources, powered by Cognition.ai 💬")
st.info("This is a generic demo, reach us out [website](https://www.google.com) to build customized AI solutions for your business requirements", icon="📃")
         
if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about the data sources you just uploaded!"}
    ]

# Modify the user_upload_pdf function to return True if a file is uploaded, otherwise False
def user_upload_pdf():
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        os.makedirs("tempDir", exist_ok=True)
        with open(os.path.join("tempDir", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
            print("----------- FILE UPLOADED SUCCESSFULLY ---------------")
        st.success("File uploaded successfully.")
        return True
    return False

# Call user_upload_pdf and store the return value
file_uploaded = user_upload_pdf()

# Execute the rest of the code only if a file has been uploaded
if file_uploaded:
    @st.cache_resource(show_spinner=False)
    def load_data():
        with st.spinner(text="Loading and indexing your documents – hang tight! This should take 1-2 minutes."):
            # The st.file_uploader() call has been moved outside of the load_data() function.
            # Please refer to the updated code outside this function for the file upload logic.
            reader = SimpleDirectoryReader(input_dir='./tempDir')
            docs = reader.load_data()
            # llm = OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="You are an expert o$
            # index = VectorStoreIndex.from_documents(docs)
            service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.1, 
                                                                      system_prompt="You are a Financial expert and your job is to answer financial questions. Assume that all questions are related to various funds, their performance and their policies. Keep your answers technical and based on facts – do not hallucinate features."))
            index = VectorStoreIndex.from_documents(docs, service_context=service_context)
            return index

    index = load_data()

    if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
            st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

    if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

    for message in st.session_state.messages: # Display the prior chat messages
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # If last message is not from assistant, generate a new response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("....."):
                response = st.session_state.chat_engine.chat(prompt)
                st.write(response.response)
                message = {"role": "assistant", "content": response.response}
                st.session_state.messages.append(message) # Add response to message history

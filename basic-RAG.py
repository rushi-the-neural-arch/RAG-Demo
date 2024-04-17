import os, streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, StorageContext, load_index_from_storage
import traceback

from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.indices.postprocessor import SimilarityPostprocessor

from dotenv import load_dotenv
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

# Define a simple Streamlit app
st.title("knowledge Extraction from various data sources, powered by CognitionAI 💬")

def user_upload_pdf():
    if 'pdf_uploaded' not in st.session_state or not st.session_state.pdf_uploaded:
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        if uploaded_file is not None:
            os.makedirs("tempDir", exist_ok=True)
            with open(os.path.join("tempDir", uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
                print("----------- FILE UPLOADED SUCCESSFULLY ---------------")
            st.success("File uploaded successfully.")
            st.session_state.pdf_uploaded = True
    else:
        st.success("File already uploaded.")

def load_or_create_index():
    PERSIST_DIR = "./storage"
    with st.spinner('Creating or loading index, please wait...'):
        if 'index_loaded' not in st.session_state or not st.session_state.index_loaded:
            if not os.path.exists(PERSIST_DIR):
                print("Creating index and saving them locally")
                documents = SimpleDirectoryReader(input_dir='./tempDir').load_data()
                index = VectorStoreIndex.from_documents(documents, show_progress=True)
                index.storage_context.persist(persist_dir=PERSIST_DIR)
            else:
                print("Loading embeddings from disk")
                storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
                index = load_index_from_storage(storage_context)
            st.session_state.index = index
            st.session_state.index_loaded = True
        else:
            print("Using cached index")
            index = st.session_state.index
    return index

def process_query(index):
    query = st.text_input("What would you like to ask?")
    if st.button("Submit"):
        if not query.strip():
            st.error("Please provide the search query.")
        else:
            with st.spinner('Processing your query, please wait...'):
                try:
                    retriever = VectorIndexRetriever(index, similarity_top_k=5)
                    postprocessor = SimilarityPostprocessor(similarity_cutoff=0.8)
                    query_engine = RetrieverQueryEngine(retriever, node_postprocessors=[postprocessor])

                    # role_prompt = "As an expert in the Finance field, provide detailed answers from the knowledge extracted from strictly the information provided only and do not hallucinate. "
                    # modified_query = role_prompt + query

                    response = query_engine.query(query)
                    st.success(response)

                    references = {}
                    for i in range(len(response.source_nodes)):
                        references[i] = {}

                        file_name = response.source_nodes[i].metadata.get('file_name')
                        references[i]['file_name'] = file_name

                        page = response.source_nodes[i].metadata.get('page_label')
                        references[i]['page'] = page

                        probability = response.source_nodes[0].score
                        references[i]['probability'] = probability

                        reference_text = response.source_nodes[i].text
                        references[i]['text'] = reference_text

                    st.write(references)

                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    st.error(traceback.format_exc())

# Initialize or use existing session state variables
if 'pdf_uploaded' not in st.session_state:
    st.session_state.pdf_uploaded = False
if 'index_loaded' not in st.session_state:
    st.session_state.index_loaded = False

user_upload_pdf()

if st.session_state.pdf_uploaded:
    index = load_or_create_index()
    process_query(index)
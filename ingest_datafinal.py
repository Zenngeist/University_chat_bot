import os
import pickle
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime
import chromadb # Import chromadb

# --- NEW: Import the ingestion logic directly ---
import ingestion

# --- Imports for Advanced Retrievers ---
from langchain.retrievers import ParentDocumentRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Core LangChain Imports ---
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage

# --- File Paths (Now relative for portability) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_STORE_PATH = os.path.join(SCRIPT_DIR, "vectorstore")
DOC_STORE_FILE_PATH = os.path.join(SCRIPT_DIR, "docstore.pkl")

# --- Auto-setup Logic for Deployment ---
# This checks if the knowledge base exists on the server. If not, it builds it once.
if not os.path.exists(VECTOR_STORE_PATH) or not os.path.exists(DOC_STORE_FILE_PATH):
    st.info("Knowledge base not found. Building it for the first time...")
    st.warning("This is a one-time setup and may take several minutes. Please be patient.")
    
    with st.spinner("Processing documents and creating vector store..."):
        ingestion.build_knowledge_base()
    
    st.success("Knowledge base built successfully! The app will now load.")
    st.button("Start Chatbot")

@st.cache_resource
def load_advanced_rag_chain():
    """
    Loads all components for the RAG chain. This is cached for performance.
    """
    print("--- Running FINAL Multi-Query RAG Chain Setup ---")

    load_dotenv()
    # Securely load the API key for both local and server environments
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise ValueError("Google API key not found. Please set it in .env or Streamlit secrets.")

    print(f"Loading parent document store from: {DOC_STORE_FILE_PATH}")
    try:
        with open(DOC_STORE_FILE_PATH, "rb") as f:
            raw_docstore = pickle.load(f)
        store = InMemoryStore()
        store.store = raw_docstore
        print("  ✓ Parent document store loaded successfully.")
    except FileNotFoundError:
        raise FileNotFoundError(f"'{DOC_STORE_FILE_PATH}' not found. Ensure ingestion was successful.")

    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GOOGLE_API_KEY
    )
    
    # --- FIX: Explicitly connect to the persistent ChromaDB client for stability ---
    print(f"Loading vector store from: {VECTOR_STORE_PATH}")
    chroma_client = chromadb.PersistentClient(path=VECTOR_STORE_PATH)
    vector_store = Chroma(
        client=chroma_client,
        collection_name="parent_document_retrieval", # Must match ingestion script
        embedding_function=embedding_model
    )
    
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

    base_retriever = ParentDocumentRetriever(
        vectorstore=vector_store,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )
    print("  ✓ Base ParentDocumentRetriever reconstructed successfully.")

    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash", 
        temperature=0.3,
        google_api_key=GOOGLE_API_KEY
    )

    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever, llm=llm
    )
    print("  ✓ Multi-Query Retriever is ready.")

    condense_question_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_question_template)

    qa_prompt_template = """You are a helpful and precise university assistant chatbot for DIT University. 
    Your goal is to provide accurate and detailed answers. Formulate your answers in natural sentences and a helpful tone.

    **Instructions:**
    1. For questions about university facts (schedules, rules, dates, fees, grades, faculty), you MUST base your answer ONLY on the provided context.
    2. Do not just copy-paste from the context. You must synthesize the information from the context and rephrase it into your own clear and well-structured sentences.
    3. If the factual information is not in the context, explicitly say "I searched through all available documents, but I could not find information on that topic." and then suggest checking the official DIT University website (https://www.dituniversity.edu.in/) or the student ERP portal (https://diterp.dituniversity.edu.in). Do not make up factual information.
    4. For academic questions from the curriculum (e.g., "solve this numerical," "explain this algorithm," "write this code"), first try to answer using the context. If the context is insufficient to solve the problem, you are then allowed to use your own general knowledge to provide a helpful, educational answer.

    Context:
    {context}

    Question: {question}

    Helpful Answer:
    """
    QA_PROMPT = PromptTemplate(template=qa_prompt_template, input_variables=["context", "question"])
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=multi_query_retriever,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True,
    )
    print("--- RAG Chain Setup Complete ---")
    return qa_chain

# --- UI and Main App Logic ---
def process_query(qa_chain, prompt):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.spinner("Thinking..."):
        try:
            response = qa_chain.invoke({
                "question": prompt,
                "chat_history": st.session_state.chat_history
            })
            answer = response["answer"]

            st.session_state.messages.append({"role": "assistant", "content": answer})
            
            st.session_state.chat_history.append(HumanMessage(content=prompt))
            st.session_state.chat_history.append(AIMessage(content=answer))
        except Exception as e:
            error_message = f"Oof, something went wrong: {e}"
            st.session_state.messages.append({"role": "assistant", "content": error_message})

st.set_page_config(page_title="DIT University AI Assistant", page_icon="🎓")
st.title("🎓 DIT University AI Assistant")

qa_chain = load_advanced_rag_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if not st.session_state.messages:
    st.session_state.messages.append({"role": "assistant", "content": "Hey there, I'm here to help you with anything DIT University related. What's on your mind?"})
    
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if len(st.session_state.messages) <= 1:
    today_name = datetime.now().strftime('%A')
    
    st.write("Here are some things you can ask:")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🗓️ When is Youthopia fest?"):
            process_query(qa_chain, "When is Youthopia fest?")
            st.rerun() 
    with col2:
        if st.button("✍️ What is the Mid-term schedule?"):
            process_query(qa_chain, "What is the Mid-term schedule?")
            st.rerun()
    with col3:
        if st.button(f"Today's Time Table ({today_name})"):
            process_query(qa_chain, f"What is my full schedule for {today_name}?")
            st.rerun()

if prompt := st.chat_input("Ask me something about the university..."):
    process_query(prompt)
    st.rerun()

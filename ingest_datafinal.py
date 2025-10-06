import os
import json
import pickle
from python-dotenv import load_dotenv
from datetime import datetime

# --- Loader Imports ---
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, Docx2txtLoader

# --- Retriever Imports ---
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Core LangChain Imports ---
from langchain.docstore.document import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

print("--- Starting Final Advanced Ingestion Script ---")

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Google API key not found. Please set it in the .env file.")
else:
    print("✓ API Key loaded successfully.")

# --- Define Paths using relative paths for portability ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "data")
VECTOR_STORE_PATH = os.path.join(SCRIPT_DIR, "vectorstore")
DOC_STORE_FILE_PATH = os.path.join(SCRIPT_DIR, "docstore.pkl")
all_documents = []

def get_document_metadata(filename):
    """
    Assigns highly granular topics to each document for precise retrieval.
    """
    base_name, _ = os.path.splitext(filename)
    normalized_name = base_name.lower().replace('_', ' ').replace('-', ' ').replace('.', ' ')
    
    topic = "general_university_info" # Default topic
    if "time table" in normalized_name or "timetable" in normalized_name:
        topic = "student_timetable"
    elif "performance" in normalized_name or "indices" in normalized_name or "grade" in normalized_name:
        topic = "student_performance_indices"
    elif "attendance" in normalized_name:
        topic = "attendance_report"
    elif "khaana" in normalized_name or "mess menu" in normalized_name:
        topic = "mess_menu"
    elif "library" in normalized_name:
        topic = "library_info"
    elif "clubs" in normalized_name:
        topic = "student_clubs"
    elif "examination schedule" in normalized_name:
        topic = "specific_exam_schedule"
    elif "calendar" in normalized_name:
        topic = "academic_calendar"
    elif "ordinance" in normalized_name or "conduct" in normalized_name or "hand book" in normalized_name:
        topic = "rules_and_regulations"
    elif "faculty" in normalized_name or "instructor" in normalized_name:
        topic = "faculty_info"
    elif "fee" in normalized_name or "scholarship" in normalized_name:
        topic = "admissions_and_fees"
    elif "syllabus" in normalized_name or "btcse batch" in normalized_name:
        topic = "course_syllabus"
        
    else:
        subject_found = None
        if "advanced java" in normalized_name:
            subject_found = "advanced_java"
        elif "r programming" in normalized_name:
            subject_found = "r_programming"
        elif "machine learning" in normalized_name:
            subject_found = "machine_learning"
        elif "artificial intelligence" in normalized_name or "ai" in normalized_name:
            subject_found = "artificial_intelligence"
        elif "computer networks" in normalized_name:
            subject_found = "computer_networks"
        elif "product design" in normalized_name:
            subject_found = "product_design"
        
        else:
            if any(kw in normalized_name for kw in ["csn302", "caf612", "csf206"]):
                subject_found = "advanced_java"
            elif any(kw in normalized_name for kw in ["csn341", "csf341", "it345"]):
                subject_found = "r_programming"
            elif any(kw in normalized_name for kw in ["csn344", "csf344", "cs402", "cs401", "csf382", "mef453"]):
                subject_found = "machine_learning"
            elif any(kw in normalized_name for kw in ["ca312", "csn304", "csf304", "ib343", "csf611"]):
                subject_found = "artificial_intelligence"
            elif any(kw in normalized_name for kw in ["csn303", "caf206", "cs303d", "csf303", "modcaf701", "cs348d", "csf351"]):
                subject_found = "computer_networks"
            elif any(kw in normalized_name for kw in ["men446", "mef446"]):
                subject_found = "product_design"
        
        if subject_found:
            is_qp = any(kw in normalized_name for kw in ["qp", "mid term", "end term", "mte", "back", "examination", "paper"])
            topic_prefix = "exam_paper_" if is_qp else "course_material_"
            topic = topic_prefix + subject_found
        elif any(kw in normalized_name for kw in ["qp", "mid term", "end term", "mte", "back", "examination", "paper"]):
             topic = "exam_paper_other"
            
    return {"source": filename, "topic": topic, "ingestion_date": datetime.now().strftime("%Y-%m-%d")}

print(f"\n--- Stage 1: Loading & Enriching Documents from '{DATA_PATH}' ---")
try:
    if not os.path.isdir(DATA_PATH):
        raise FileNotFoundError(f"The directory '{DATA_PATH}' was not found.")
    for filename in os.listdir(DATA_PATH):
        file_path = os.path.join(DATA_PATH, filename)
        metadata = get_document_metadata(filename)
        print(f"  > Found '{filename}' | Topic: {metadata['topic']}")
        
        try:
            if filename.endswith('.pdf'): 
                loader = PyMuPDFLoader(file_path)
            elif filename.endswith('.docx'): 
                loader = Docx2txtLoader(file_path)
            elif filename.endswith('.txt'): 
                loader = TextLoader(file_path, encoding='utf-8')
            else:
                print(f"    > Skipping unsupported file: {filename}")
                continue
            
            docs = loader.load()
            for doc in docs: 
                doc.metadata.update(metadata)
            all_documents.extend(docs)
        except Exception as e:
            print(f"    !!! WARNING: Failed to load {filename}. Error: {e}")
except Exception as e:
    print(f"!!! ERROR in Stage 1: {e}")

print(f"\n--- CHECKPOINT: Total documents loaded: {len(all_documents)} ---")

if not all_documents:
    print("!!! CRITICAL ERROR: No documents were loaded.")
else:
    print("\n--- Stage 2: Setting up Advanced Retriever ---")
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)
    print("  > Embedding model initialized.")
    
    vector_store = Chroma(collection_name="parent_document_retrieval", embedding_function=embedding_model, persist_directory=VECTOR_STORE_PATH)
    store = InMemoryStore()
    print("  > Stores initialized.")
    
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    
    retriever = ParentDocumentRetriever(vectorstore=vector_store, docstore=store, child_splitter=child_splitter, parent_splitter=parent_splitter)
    print("  > ParentDocumentRetriever is ready.")
    
    print("\n--- Stage 3: Adding Documents in Batches (This will take time) ---")
    batch_size = 100
    total_docs = len(all_documents)
    
    for i in range(0, total_docs, batch_size):
        batch = all_documents[i : i + batch_size]
        print(f"  > Processing batch {i//batch_size + 1}/{(total_docs + batch_size - 1)//batch_size}...")
        retriever.add_documents(batch, ids=None)

    print("  > Data ingestion complete.")
    
    print(f"\n--- Stage 4: Saving parent document store to {DOC_STORE_FILE_PATH} ---")
    try:
        with open(DOC_STORE_FILE_PATH, "wb") as f: 
            pickle.dump(store.store, f)
        print("  ✓ Parent document store saved successfully.")
    except Exception as e:
        print(f"!!! ERROR in Stage 4: {e}")
        
    print(f"\n  ✓ Vector store persisted at: {VECTOR_STORE_PATH}")

print("\n--- Ingestion Script Finished ---")


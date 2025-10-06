import os
import pickle
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime

from langchain.retrievers import ParentDocumentRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage


@st.cache_resource
def load_advanced_rag_chain():
    """
    Loads all components for the RAG chain, now featuring the Multi-Query Retriever.
    """
    print("running FINAL Multi-Query RAG Chain Setup")

    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise ValueError("Google API key not found. Make sure you have a .env file with GOOGLE_API_KEY set.")

    VECTOR_STORE_PATH = "/home/zengeist/Documents/university-chatbot/vectorstore"
    DOC_STORE_FILE_PATH = "/home/zengeist/Documents/university-chatbot/docstore.pkl"

    try:
        with open(DOC_STORE_FILE_PATH, "rb") as f:
            raw_docstore = pickle.load(f)
        store = InMemoryStore(); store.store = raw_docstore
        print("  ✓ Parent document store loaded.")
    except FileNotFoundError:
        raise FileNotFoundError(f"'{DOC_STORE_FILE_PATH}' not found. Please run the final ingest_data.py script first.")

    embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)
    vector_store = Chroma(collection_name="final_retrieval_system", persist_directory=VECTOR_STORE_PATH, embedding_function=embedding_model)

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

    base_retriever = ParentDocumentRetriever(vectorstore=vector_store, docstore=store, child_splitter=child_splitter, parent_splitter=parent_splitter)
    print("  ✓ Base ParentDocumentRetriever reconstructed.")

    llm_for_queries = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0, google_api_key=GOOGLE_API_KEY)

    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever, 
        llm=llm_for_queries
    )
    print("  ✓ Multi-Query Retriever is ready.")

    condense_question_template = (
        "Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.\n\n"
        "Chat History:\n{chat_history}\nFollow Up Input: {question}\nStandalone question:"
    )
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_question_template)

    qa_prompt_template = """
You are a helpful and precise university assistant chatbot for DIT University.

- Use ONLY the information in the provided Context when answering factual queries about schedules, policies, dates, events, timetables, or any document-specific data.
- Do NOT invent factual claims that are not present in the Context.
- When you use the Context, DO NOT copy long sentences verbatim. Rephrase and paraphrase the content in your own words while preserving meaning. Avoid copying sequences of more than 8 consecutive words from the Context.

- If the user's question is a problem-solving request (numerical calculations, programming exercises, derivations, or other curriculum-style problems), you SHOULD attempt to solve it yourself using your internal knowledge and reasoning. If the Context provides a problem statement, constraints, or data, incorporate them. If the Context does not contain the problem, you may compute the answer independently.

- For problem-solving answers, show brief step-by-step reasoning or the main steps you used to arrive at the result (concise, 3-6 lines is fine).

- If you cannot find the factual information in Context, reply exactly:
"I searched through all available documents, but I could not find information on that topic. Please check the official DIT University website or the ERP portal."

Context:
{context}

Question: {question}

Helpful Answer:
"""
    QA_PROMPT = PromptTemplate(template=qa_prompt_template, input_variables=["context", "question"])

    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0.2, google_api_key=GOOGLE_API_KEY)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=multi_query_retriever,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True,
    )
    print("RAG Chain Setup Complete")
    return qa_chain


st.set_page_config(page_title="DIT University AI Assistant", page_icon="🎓")
st.title(" DIT University AI Assistant")
qa_chain = load_advanced_rag_chain()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hey there, I'm here to help you with anything DIT University related. What's on your mind?"}]
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "kickstarter_shown" not in st.session_state:
    st.session_state.kickstarter_shown = True


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.kickstarter_shown:
    st.markdown("**Quick actions — click to auto-run a query**")
    kickstarter_questions = [
        ("What is the Mid term schedule?", "What is the Mid term schedule?"),
        ("When is Youthopia fest?", "When is Youthopia fest?"),
        ("Time Table for Today", f"What is the timetable for {datetime.now().strftime('%A')}?")
    ]
    for label, actual_prompt in kickstarter_questions:
        if st.button(label):
            st.session_state.kickstarter_shown = False
            st.session_state.messages.append({"role": "user", "content": actual_prompt})
            with st.chat_message("user"): st.write(actual_prompt)
            try:
                with st.spinner("Searching documents..."):
                    response = qa_chain.invoke({"question": actual_prompt, "chat_history": st.session_state.chat_history})
                    answer = response["answer"]
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    with st.chat_message("assistant"): st.write(answer)
                    st.session_state.chat_history.append(HumanMessage(content=actual_prompt))
                    st.session_state.chat_history.append(AIMessage(content=answer))
            except Exception as e:
                error_message = f"Oof, something went wrong: {e}"
                st.session_state.messages.append({"role": "assistant", "content": error_message})
                with st.chat_message("assistant"): st.write(error_message)
            break  

if prompt := st.chat_input("Ask me something about the university..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.write(prompt)
    with st.spinner("Thinking..."):
        try:
            response = qa_chain.invoke({"question": prompt, "chat_history": st.session_state.chat_history})
            answer = response["answer"]
            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"): st.write(answer)
            st.session_state.chat_history.append(HumanMessage(content=prompt))
            st.session_state.chat_history.append(AIMessage(content=answer))
        except Exception as e:
            error_message = f"Oof, something went wrong: {e}"
            st.session_state.messages.append({"role": "assistant", "content": error_message})
            with st.chat_message("assistant"): st.write(error_message)

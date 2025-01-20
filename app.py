import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
##
llm = ChatGroq(groq_api_key="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", model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
    """
    Based on the provided context, return only the query directly related to the question.
    <context>
    {context}
    <context>
    Question: {input}
    """
)

def create_vector_embedding(uploaded_file):
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.loader = PyPDFLoader(uploaded_file)
        st.session_state.docs = st.session_state.loader.load()  # Document loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

st.title("RAG Document Chatbot")

uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file is not None:
    st.write("PDF uploaded successfully!")

user_prompt = st.text_input("Enter your query from the document")

if st.button("Result") and uploaded_file:
    # Save the uploaded file temporarily for processing
    with open("temp_uploaded_file.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    create_vector_embedding("temp_uploaded_file.pdf")
    st.write("Vector Database is ready")

import time

if user_prompt and uploaded_file:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retriever_chain = create_retrieval_chain(retriever, document_chain)
    
    start = time.process_time()
    response = retriever_chain.invoke({'input': user_prompt})
    print(f"Response time: {time.process_time() - start}")
    
    st.write(response['answer'])

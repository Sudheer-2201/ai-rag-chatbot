import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

load_dotenv()

st.title("AI Knowledge Chatbot")

loader = TextLoader("data/data.txt")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = FAISS.from_documents(texts, embeddings)

llm = ChatGroq(model_name="llama-3.1-8b-instant")

qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

query = st.text_input("Ask a question")

if query:
    result = qa.invoke({"query": query})["result"]
    st.write("Bot:", result)
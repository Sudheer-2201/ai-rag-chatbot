from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

load_dotenv()

app = FastAPI()

loader = TextLoader("data/data.txt")
documents = loader.load()

splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.from_documents(texts, embeddings)

llm = ChatGroq(model_name="llama-3.1-8b-instant")

qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())


class Query(BaseModel):
    question: str


@app.post("/chat")
def chat(query: Query):
    result = qa.invoke({"query": query.question})["result"]
    return {"answer": result}
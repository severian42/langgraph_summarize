from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, TextLoader, JSONLoader
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

vectorstore = None
retriever = None

def load_and_split_documents(docs):
    logger.info(f"Splitting {len(docs)} documents")
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=512, chunk_overlap=0
    )
    return text_splitter.split_documents(docs)

def initialize_vectorstore():
    global vectorstore, retriever
    
    logger.info("Initializing vectorstore")
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
    doc_splits = load_and_split_documents(docs_list)

    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
        persist_directory="./.chroma",
    )
    retriever = vectorstore.as_retriever()
    logger.info("Vectorstore initialized")

def ingest_documents(files):
    global vectorstore, retriever
    
    if vectorstore is None:
        initialize_vectorstore()
    
    new_docs = []
    for file in files:
        file_extension = os.path.splitext(file.name)[1].lower()
        logger.info(f"Processing file: {file.name}")
        if file_extension == '.pdf':
            loader = PyPDFLoader(file.name)
        elif file_extension in ['.txt', '.md']:
            loader = TextLoader(file.name)
        elif file_extension == '.json':
            loader = JSONLoader(
                file_path=file.name,
                jq_schema='.[]',
                text_content=False,
                json_lines=False
            )
        else:
            logger.error(f"Unsupported file type: {file_extension}")
            raise ValueError(f"Unsupported file type: {file_extension}")
        new_docs.extend(loader.load())
    
    logger.info(f"Loaded {len(new_docs)} new documents")
    doc_splits = load_and_split_documents(new_docs)
    logger.info(f"Adding {len(doc_splits)} document splits to vectorstore")
    vectorstore.add_documents(doc_splits)
    
    # Instead of using persist(), we'll recreate the retriever
    retriever = vectorstore.as_retriever()
    logger.info("Documents ingested and retriever updated")

def get_retriever():
    global retriever
    if retriever is None:
        initialize_vectorstore()
    return retriever

# Initialize the vectorstore when the module is imported
initialize_vectorstore()

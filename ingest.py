import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# --- Put your Pinecone API Key here ---
os.environ["PINECONE_API_KEY"] = "pinecone api key"
INDEX_NAME = "enterprise-rag"

def ingest_data():
    print("1. Loading documents from ./data folder...")
    loader = PyPDFDirectoryLoader("./data")
    docs = loader.load()
    
    print("2. Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)
    
    print("3. Loading Embedding Model (bge-small-en-v1.5) on RTX 5050...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'} 
    )
    
    print("4. Connecting to Pinecone...")
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    
    # Check if index exists. Note: BAAI/bge-small outputs 384 dimensions
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    if INDEX_NAME not in existing_indexes:
        print(f"Creating new Pinecone index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=384, # Must be 384 for this specific model
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        
    print(f"5. Uploading {len(chunks)} chunks to Pinecone cloud...")
    PineconeVectorStore.from_documents(chunks, embeddings, index_name=INDEX_NAME)
    print("Upload complete! You can now start the web server.")

if __name__ == "__main__":
    ingest_data()
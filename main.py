import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from pinecone import Pinecone

# --- API KEYS ---
os.environ["GROQ_API_KEY"] = "API key for Groq"
os.environ["PINECONE_API_KEY"] = "api key for Pinecone"
INDEX_NAME = "enterprise-rag"

app = FastAPI()

# 1. Initialize Embeddings and Vector Store
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={'device': 'cpu'} # Uses your RTX 5050
)
vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 2. Setup LLM (Groq) and Chain
llm = ChatGroq(model_name="openai/gpt-oss-20b")
system_prompt = (
    "You are an enterprise AI knowledge assistant. Answer the user's question strictly "
    "using the retrieved context below. If the answer is not in the context, say 'I don't know'. "
    "Do not hallucinate facts.\n\nContext:\n{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])
rag_chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))

# 3. Setup the API Route for the HTML page to call
class QueryRequest(BaseModel):
    query: str

@app.post("/api/query")
async def query_endpoint(req: QueryRequest):
    response = rag_chain.invoke({"input": req.query})
    
    # Format the sources to send back to the HTML
    sources = []
    for doc in response.get("context", []):
        sources.append({
            "source": doc.metadata.get("source", "Unknown Document").split("/")[-1],
            "page": doc.metadata.get("page", "N/A")
        })
        
    return {"answer": response["answer"], "sources": sources}

# 4. Serve the HTML page
@app.get("/")
async def get_ui():
    with open("index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
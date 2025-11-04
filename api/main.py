import os
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from typing import List
from dotenv import load_dotenv

# load .env from the current working directory
load_dotenv()

# Initializing FastAPI app
app = FastAPI(title="RAG Backend")

# ---- In-memory Stroage ----
vector_store = None

# --- Helper Functions ----
def get_embeddings():
    """Initializes and reutnrs the Ollama embedding model."""
    embedding_model_name = os.getenv("OLLAMA_EMBEDDING_MODEL_NAME", "")
    if not  embedding_model_name:
        raise HTTPException(status_code = 500, detail="No ollama model name found, please set OLLAMA_EMBEDDING_MODEL_NAME in .env")
    return OllamaEmbeddings(model=embedding_model_name)

def get_llm():
    """Initizlizes adn returns the Ollama LLM"""
    ollama_model_name = os.getenv("OLLAMA_MODEL_NAME", "")
    if not ollama_model_name:
        raise HTTPException(status_code = 500, detail="No ollama model name found, please set OLLAMA_MODEL_NAME in .env")
    return ChatOllama(model=ollama_model_name)

def debug_prompt(prompt):
    print("Prompt being sent to the LLM: ")
    print(prompt)
    return prompt

def create_rag_chain(retriever):
    """Create and return a RAG chain """
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(
        """ You are a helpful study assistant. Use the following peices of retrieved context to answer the question.
        If you don't know the answer, just say that you don't know. Be concise and helpful.

        Question: {question}

        Context: {context}

        Answer: 
        """
    )
    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | RunnableLambda(debug_prompt) # logs prompt before LLM call
        | llm
        | StrOutputParser()
    )

# ---- API Endpoints ----

@app.get("/hello")
def read_root():
    return{"Hello": "World"}
# GET - retrives data
# POST - sending data
# PUT - update a database
# DELETE

@app.post("/upload/")
async def upload_and_process_files(files: List[UploadFile] = File()):
    """
    Upload PDF files, processes them, and creates a vectore store.
    """
    global vector_store
    all_chunks = []

    # Ensure the 'temp' directory exists
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)

    for uploaded_file in files:
        file_path = os.path.join(temp_dir, uploaded_file.filename)
        try: 
            # Save the uploaded file temporarily
            with open(file_path, "wb") as f:
                f.write(await uploaded_file.read())

            # Load and process the PDF
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            chunk_size=int(os.getenv("CHUNK_SIZE", 1000))
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 200))
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(docs)
            all_chunks.extend(chunks)
        except Exception as e:
            # Clean up the temp file on error
            if os.path.exists(file_path):
                os.remove(file_path)
            raise HTTPException(status_code=500, detail=f"Error processing file {uploaded_file.filename}: {e}")
        finally: 
            # Clean up the temp file on error
            if os.path.exists(file_path):
                os.remove(file_path)

    if not all_chunks:
        raise HTTPException(status_code=400, detail="No content to process found in the uploaded files")
    
    print(all_chunks[:3]) # debug
    # Create embedding and vector store
    embeddings = get_embeddings()
    vector_store = Chroma.from_documents(documents=all_chunks, embedding=embeddings)

    # TO-DO: inject time info here
    return {"message" : "Files processed successfully and vector store created"}

@app.post("/ask/")
async def ask_question(question: str = Form(...)):
    """
    Asks a question to the RAG chain usign the previously created vector store.
    """
    # vector store
    # question -> embed -> search in vector store -> retrieve tp k=5 chunks -> put prompt -> send to LLM -> get response.
    global vector_store
    if not vector_store:
        raise HTTPException(status_code=400, detail="Vector store not found. Please upload documents first.")
    
    if not question: 
        raise HTTPException(status_code=400, detail="Please enter a question.")

    try:
        k = int(os.getenv("K", 4))
        retriever = vector_store.as_retriever(search_kwargs={"k": k})
        rag_chain = create_rag_chain(retriever)
        response = rag_chain.invoke(question)
        print("LLM response: ", response)
        return {"answer": response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during question answering: {e}")

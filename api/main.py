import os
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from typing import List
from dotenv import load_dotenv
import base64
# load .env from the current working directory
load_dotenv()

# Initializing FastAPI app
app = FastAPI(title="RAG Backend")

# ---- In-memory Stroage ----
vector_store = None

# --- Supported image extensions ---
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}

# --- Helper Functions ----
def get_embeddings():
    """Initializes and reutnrs the Ollama embedding model."""
    embedding_model_name = os.getenv("OLLAMA_EMBEDDING_MODEL_NAME", "")
    if not embedding_model_name:
        raise HTTPException(status_code = 500, detail="No ollama model name found, please set OLLAMA_EMBEDDING_MODEL_NAME in .env")
    return OllamaEmbeddings(model=embedding_model_name)

def get_llm():
    """Initizlizes adn returns the Ollama LLM"""
    ollama_model_name = os.getenv("OLLAMA_MODEL_NAME", "")
    if ollama_model_name:
        return ChatOllama(model=ollama_model_name)
    
    print("No Ollama model initialized, falling back to Groq...")
    groq_api_key = os.getenv("GROQ_API_KEY", "")
    groq_model_name = os.getenv("GROQ_MODEL_NAME", "")
    if groq_api_key and groq_model_name:
        return ChatGroq(
            model_name=groq_model_name,
            groq_api_key=groq_api_key
        )
        

    
    raise HTTPException(status_code = 500, detail="No ollama model name found or Groq model name found + API key found, please set in .env")    

def debug_prompt(prompt):
    print("Prompt being sent to the LLM: ")
    print(prompt)
    return prompt

def create_rag_chain(retriever):
    """Create and return a RAG chain """
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(
        """ You are a helpful study assistant.
            
            Use the retrieved context as your primary source when answering the question, but you may also rely on your general knowledge if the context is incomplete.
            If you use information not found in the retrieved context, ensure it is well-established and widely accepted—not speculative.
            If you are unsure or the information is not reliable, say “I'm not sure” or “The context does not provide enough information.”

            Be concise, accurate, and helpful.

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

def encode_image_base64(image_path: str, ext: str) -> str:
    
    with open(image_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    return f"data:image/{ext};base64,{b64}"
                                                            

def caption_image(file_path: str, ext: str) -> str:
    """
    Takes an image file pat hand returns a caption only using a vision-enabled LLM
    """
    if isinstance(file_path, str):
        print("image file path: ", file_path)
    ollama_vision_model_name = os.getenv("OLLAMA_VISION_MODEL_NAME",)
    groq_api_key = os.getenv("GROQ_API_KEY", "")
    groq_vision_model_name = os.getenv("GROQ_VISION_MODEL_NAME", "")
    if ollama_vision_model_name:
        llm = ChatOllama(model=ollama_vision_model_name)
    elif groq_vision_model_name and groq_api_key:
        llm = ChatGroq(
            model_name = groq_vision_model_name,
            groq_api_key = groq_api_key
        )
    else:
        raise ValueError("No vision model could be found. Please set in .env")

    image_url = encode_image_base64(image_path=file_path, ext=ext)
    msg = HumanMessage(
        content=[
            {"type": "text", "text": "Describe this image in detail so it can be used as searchable knowledge"},
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                    "detail":"auto"
                    }
                }
        ]
    )

    result = llm.invoke([msg])
    return result.content


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
            
            ext = os.path.splitext(uploaded_file.filename)[1].lower()

            # ---1) PDF CASE ---
            if ext =='.pdf':

                # Load and process the PDF
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                chunk_size=int(os.getenv("CHUNK_SIZE", 1000))
                chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 200))
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                chunks = text_splitter.split_documents(docs)
                all_chunks.extend(chunks)

            # ---2) IMAGE CASE ---
            elif ext in IMAGE_EXTS:
                print("Processing image: ", uploaded_file.filename)
                caption = caption_image(file_path = file_path, ext=ext)
                print("Image caption: ", caption)
                img_doc = Document(
                    page_content=caption,
                    metadata={"source" : uploaded_file.filename, "type": "image"}

                )
                all_chunks.append(img_doc)
            else:
                raise ValueError(f"Unsupported file type: {ext}")
            
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

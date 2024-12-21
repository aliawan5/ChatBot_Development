from fastapi import FastAPI, UploadFile, HTTPException
import logging
from fastapi.responses import JSONResponse
from langchain_community.document_loaders import PyPDFLoader
from rag import RagPipeline
from pydantic import BaseModel

app = FastAPI()

pipeline = None

@app.post("/upload")
async def upload_document(file: UploadFile):
    """
    Endpoint to upload a document, load it using PyPDFLoader, and initialize the pipeline.
    """
    global pipeline

    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    try:
        file_content = await file.read()
        loader = PyPDFLoader(file_content)
        doc = loader.load()
        logging.info("Document Loaded Successfully")

        if not pipeline:
            pipeline = RagPipeline(document=file.filename, model_name="ollama-model", prompt="")
            pipeline.doc = doc
            pipeline.create_chunks()
            pipeline.create_db()
        else:
            logging.info("Pipeline already initialized. Skipping reinitialization.")

        return JSONResponse({"message": "Document uploaded and processed successfully."})

    except Exception as e:
        logging.error(f"Error processing document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


class ChatRequest(BaseModel):
    question: str


@app.post("/chat")
async def chat_with_bot(chat_request: ChatRequest):
    """
    Endpoint to chat with the bot using the uploaded document.
    """
    global pipeline

    try:
        if not pipeline:
            return JSONResponse({"error": "No document uploaded yet."}, status_code=400)
        
        pipeline.create_ChatPrompt_Template()
        pipeline.load_model()
        pipeline.stuff_doc_chain()
        pipeline.Create_retrieval_chain()

        response = pipeline.retrieval_chain.run(chat_request.question)
        return JSONResponse({"answer": response})

    except Exception as e:
        logging.error(f"Error during chat interaction: {str(e)}")
        return JSONResponse({"error": str(e)}, status_code=500)

import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.llms import Ollama
import logging

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
langchain_api_key = os.getenv('LANGCHAIN_API_KEY')

class RagPipeline:
    def __init__(self, document:str, model_name:str, prompt:str):
        self.document = document
        self.model_name = model_name
        self.prompt = prompt
        self.chunk = None



    def create_chunks(self):
        """
        Splits the document into chunks using RecursiveCharacterTextSplitter.
        """
        if self.doc is not None:
            try:
                logging.info("Creating Chunks of Data")
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                self.chunk = text_splitter.split(self.doc)
                return self.chunk
            
            except Exception as e:
                logging.error(f"An error occured: {str(e)}")

        else:
            return None
        
    

    def create_db(self):
        """
        Creates a vector store from the given document chunks using Chroma and OllamaEmbeddings.
        """
        if self.chunk is not None:
            try:
                logging.info("Creating database to store vectors")
                db = Chroma.from_documents(self.chunk[:10], OllamaEmbeddings())
                self.retriver = db.as_retriever()
                return self.retriver
            
            except Exception as e:
                logging.error(f"An error occured: {str(e)}")

        else:
            return None
        


    def create_ChatPrompt_Template(self):
        """
        using the chat prompt template
        """
        try:
            self.prompt = ChatPromptTemplate.from_template("""
            Answer the following question and I will give you a reward:
            <context>
            {context}
            </context>
            Question: {input}""")
            return self.prompt

        except Exception as e:
            logging.error(f"Error creating chat prompt: {e}")
            return None
    


    def load_model(self):
        """
        We will be using Ollama model
        """
        try:
            logging.info("Loading the Model")
            self.model = Ollama(model=self.model_name)
            return self.model
        
        except Exception as e:
            logging.error(f"An error occured: {str(e)}")



    def stuff_doc_chain(self):
        """
        Creating Stuff Document chaintakes all the retrieved documents
        stuff them into the context window of the language model
        and then generates a response
        """
        if self.prompt and self.model is not None:
            try:
                logging.info("Creating Stuff Document Chain")
                self.document_chain = create_stuff_documents_chain(self.model, self.prompt)
                return self.document_chain
            
            except Exception as e:
                logging.error(f"An error occured: {str(e)}")

        else:
            logging.error("Some issue with model or prompt")



    def Create_retrieval_chain(self):
        """
        Now we create the retrieval chain
        """
        try:
            logging.info("Creating Retrieval Document Chain")
            self.retrieval_chain = create_retrieval_chain(self.retriver, self.stuff_doc_chain)
            return self.retrieval_chain
        
        except Exception as e:
            logging.error(f"An error occured: {str(e)}")


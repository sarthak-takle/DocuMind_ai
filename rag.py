import os
from typing import List, Dict, Any, Optional, Union, cast
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain.schema import Document

load_dotenv()

class RAG:
    """Retrieval Augmented Generation for question answering."""
    
    def __init__(self, vector_store, temperature: float = 0.7, api_key: Optional[str] = None):
        """
        Initialize the RAG system.
        
        Args:
            vector_store: Vector store instance
            temperature: Model temperature for response generation
            api_key: Optional OpenAI API key
        """
        self.vector_store = vector_store
        self.temperature = temperature
        effective_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = None
        if effective_key:
            try:
                self.model = ChatOpenAI(temperature=temperature, api_key=effective_key)
            except Exception:
                self.model = None
                
    def set_api_key(self, api_key: str):
        """Configure or update the OpenAI API key."""
        if api_key:
            self.model = ChatOpenAI(temperature=self.temperature, api_key=api_key)
        
    def generate_answer(self, query: str, k: int = 4) -> str:
        """
        Generate an answer to a query using RAG.
        
        Args:
            query: User query
            k: Number of top documents to retrieve
            
        Returns:
            Generated answer
        """
        if not self.model:
            key = os.getenv("OPENAI_API_KEY")
            if key:
                try:
                    self.model = ChatOpenAI(temperature=self.temperature, api_key=key)
                except Exception as e:
                    return f"Error initializing OpenAI model: {str(e)}"
            else:
                return "⚠️ OpenAI API key not found. Please provide your OpenAI API key in the sidebar or configure it in the .env file."

        # Retrieve relevant documents
        docs = self.vector_store.similarity_search(query, k=k)
        
        if not docs:
            return "No relevant information found. Please try a different question or upload documents."
        
        # Format context from documents
        context = "\n\n".join([f"Document: {doc.metadata.get('source', 'Unknown')}, Page: {doc.metadata.get('page', 'Unknown')}\n{doc.page_content}" for doc in docs])
        
        # Create prompt for the model
        messages = [
            SystemMessage(content=f"You are a helpful assistant that answers questions based on the provided documents. Use only the information in the documents to answer the question. If the answer cannot be found in the documents, say so directly.\n\nDocuments:\n{context}"),
            HumanMessage(content=query)
        ]
        
        # Generate response
        try:
            response = self.model.invoke(messages)
            if isinstance(response, BaseMessage):
                return cast(str, response.content)
            return cast(str, response)
        except Exception as e:
            return f"Error generating response: {str(e)}"
"""
RAG (Retrieval-Augmented Generation) system for therapeutic chatbot.
Handles PDF processing, embedding generation, and context retrieval.
"""

import os
import json
from typing import List, Dict
from pathlib import Path
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from sqlalchemy.orm import Session
from database import KnowledgeDocument, engine, SessionLocal
import logging

logger = logging.getLogger(__name__)


class TherapeuticRAG:
    """RAG system for retrieving therapeutic knowledge from PDF documents."""
    
    def __init__(self, openai_api_key: str):
        """Initialize RAG system with OpenAI client."""
        self.client = OpenAI(api_key=openai_api_key)
        self.embedding_model = "text-embedding-ada-002"
        self.embedding_dimension = 1536
        
        # Text splitter for chunking documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def create_embedding(self, text: str) -> List[float]:
        """Create embedding for given text using OpenAI."""
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.embedding_model
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            raise
    
    def load_pdf_documents(self, pdf_directory: str = "knowledge_base"):
        """Load and process all PDF documents in the directory."""
        pdf_path = Path(pdf_directory)
        if not pdf_path.exists():
            logger.warning(f"PDF directory {pdf_directory} does not exist")
            return []
        
        all_chunks = []
        pdf_files = list(pdf_path.glob("*.pdf"))
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        for pdf_file in pdf_files:
            try:
                logger.info(f"Processing {pdf_file.name}...")
                loader = PyPDFLoader(str(pdf_file))
                documents = loader.load()
                
                # Split documents into chunks
                chunks = self.text_splitter.split_documents(documents)
                
                for i, chunk in enumerate(chunks):
                    all_chunks.append({
                        "source_file": pdf_file.name,
                        "chunk_index": i,
                        "content": chunk.page_content,
                        "doc_metadata": json.dumps(chunk.metadata)
                    })
                
                logger.info(f"Created {len(chunks)} chunks from {pdf_file.name}")
            
            except Exception as e:
                logger.error(f"Error processing {pdf_file.name}: {e}")
                continue
        
        return all_chunks
    
    def index_documents(self, db: Session, chunks: List[Dict]):
        """Create embeddings and store in database."""
        logger.info(f"Indexing {len(chunks)} document chunks...")
        
        for i, chunk in enumerate(chunks):
            try:
                # Create embedding
                embedding = self.create_embedding(chunk["content"])
                
                # Store in database
                doc = KnowledgeDocument(
                    source_file=chunk["source_file"],
                    chunk_index=chunk["chunk_index"],
                    content=chunk["content"],
                    embedding=embedding,
                    doc_metadata=chunk["doc_metadata"]
                )
                db.add(doc)
                
                if (i + 1) % 10 == 0:
                    db.commit()
                    logger.info(f"Indexed {i + 1}/{len(chunks)} chunks")
            
            except Exception as e:
                logger.error(f"Error indexing chunk {i}: {e}")
                continue
        
        db.commit()
        logger.info("Document indexing complete!")
    
    def retrieve_relevant_context(self, db: Session, query: str, k: int = 5) -> List[str]:
        """Retrieve most relevant context chunks for a query."""
        try:
            # Create embedding for query
            query_embedding = self.create_embedding(query)
            
            # Search for similar documents using pgvector
            results = db.query(KnowledgeDocument).order_by(
                KnowledgeDocument.embedding.cosine_distance(query_embedding)
            ).limit(k).all()
            
            # Extract content from results
            contexts = [doc.content for doc in results]
            
            logger.info(f"Retrieved {len(contexts)} relevant context chunks")
            return contexts
        
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return []
    
    def build_prompt_with_context(self, user_message: str, contexts: List[str], 
                                  conversation_history: List[Dict] = None) -> str:
        """Build a prompt with retrieved context and conversation history."""
        
        # System prompt for therapeutic chatbot
        system_prompt = """You are a compassionate digital wellness therapist helping people achieve happiness and well-being through technology balance. You draw from Christian Dominique's "Beyond Happy" and "The Four Aces" frameworks.

Your approach integrates:
- The Four Aces: Awareness, Acceptance, Appreciation, and Awe
- The 7Cs: Contentment, Curiosity, Creativity, Compassion, Compersion, Courage, Connection
- The 8Ps: Presence, Positivity, Purpose, Peace, Playfulness, Passion, Patience, Perseverance
- Mindfulness, Stoicism, and positive psychology principles
- Focus on internal locus of control and mindset shifts

Core Philosophy:
- Happiness is a way of being, not a destination
- The stories we tell ourselves shape our reality (narrative self)
- Present-moment awareness and equanimity reduce suffering
- Embracing challenges leads to growth
- Connection and compassion foster well-being

Guidelines:
- Maintain a warm, non-judgmental, empathetic tone
- Use Socratic questioning to promote self-discovery
- Help users shift from external to internal focus
- Guide them to recognize what they can control (dichotomy of control)
- Encourage gratitude, appreciation, and finding awe in daily life
- Keep responses to 2-3 sentences for focus and clarity
- Ask one thoughtful, open-ended question at a time
- Celebrate small wins and encourage self-compassion

Safety:
- If someone mentions self-harm or suicide, provide crisis resources immediately
- Maintain professional boundaries
- Recommend professional help for serious mental health concerns

Use the provided knowledge base context to inform your responses with the Four Aces, 7Cs, and 8Ps frameworks."""
        
        # Build context section
        context_section = "\n\n=== RELEVANT KNOWLEDGE BASE ===\n"
        for i, context in enumerate(contexts, 1):
            context_section += f"\n[Context {i}]\n{context}\n"
        
        # Build conversation history section
        history_section = ""
        if conversation_history:
            history_section = "\n\n=== CONVERSATION HISTORY ===\n"
            for msg in conversation_history[-6:]:  # Last 6 messages for context
                role = "User" if msg["role"] == "user" else "Therapist"
                history_section += f"{role}: {msg['content']}\n"
        
        # Build full prompt
        full_prompt = f"""{system_prompt}

{context_section}
{history_section}

=== CURRENT MESSAGE ===
User: {user_message}

Therapist:"""
        
        return full_prompt


def initialize_knowledge_base():
    """Initialize knowledge base by loading and indexing PDFs."""
    logger.info("Starting knowledge base initialization...")
    
    # Get OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key or openai_api_key == "your_openai_api_key_here":
        logger.warning("OpenAI API key not set. Skipping knowledge base initialization.")
        return False
    
    # Initialize RAG system
    rag = TherapeuticRAG(openai_api_key)
    
    # Create database session
    db = SessionLocal()
    
    try:
        # Check if knowledge base already exists
        existing_docs = db.query(KnowledgeDocument).count()
        if existing_docs > 0:
            logger.info(f"Knowledge base already initialized with {existing_docs} documents")
            return True
        
        # Load PDF documents
        chunks = rag.load_pdf_documents()
        
        if not chunks:
            logger.warning("No PDF documents found to index")
            return False
        
        # Index documents
        rag.index_documents(db, chunks)
        
        logger.info("Knowledge base initialization complete!")
        return True
    
    except Exception as e:
        logger.error(f"Error initializing knowledge base: {e}")
        return False
    
    finally:
        db.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize knowledge base
    initialize_knowledge_base()

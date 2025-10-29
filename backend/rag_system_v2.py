"""
RAG (Retrieval-Augmented Generation) system for therapeutic chatbot - MVP VERSION.
Handles PDF processing, embedding generation, context retrieval with citations.

ENHANCEMENTS:
- Enhanced metadata with book titles, page numbers, timestamps
- Citability support with source tracking
- Robust error handling with graceful degradation
- Detailed logging and metrics
"""

import os
import json
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from openai import OpenAI, OpenAIError, RateLimitError, APIError
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from sqlalchemy.orm import Session
from database import KnowledgeDocument, engine, SessionLocal
import logging
from datetime import datetime
import time

logger = logging.getLogger(__name__)

# Book metadata mapping
BOOK_METADATA = {
    "4As_Manuscript_v6.pdf": {
        "title": "The Four Aces: Awakening to Happiness",
        "author": "Christian Dominique",
        "content_type": "therapeutic_framework",
        "topics": ["awareness", "acceptance", "appreciation", "awe", "mindfulness"]
    },
    "BeyondHappy_MANUSCRIPT_v7.pdf": {
        "title": "Beyond Happy: Formulas for Perfect Days",
        "author": "Christian Dominique",
        "content_type": "therapeutic_framework",
        "topics": ["7Cs", "8Ps", "stoicism", "positive_psychology", "happiness"]
    }
}


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
    
    def create_embedding(self, text: str, retry_count: int = 3) -> Optional[List[float]]:
        """
        Create embedding for given text using OpenAI with retry logic.
        
        Args:
            text: Text to embed
            retry_count: Number of retries on failure
        
        Returns:
            Embedding vector or None on failure
        """
        for attempt in range(retry_count):
            try:
                response = self.client.embeddings.create(
                    input=text,
                    model=self.embedding_model
                )
                return response.data[0].embedding
            
            except RateLimitError as e:
                wait_time = (attempt + 1) * 2  # Exponential backoff
                logger.warning(f"⚠️ Rate limit hit, waiting {wait_time}s before retry {attempt + 1}/{retry_count}")
                time.sleep(wait_time)
            
            except APIError as e:
                logger.error(f"❌ OpenAI API error: {e}")
                if attempt < retry_count - 1:
                    time.sleep(1)
                else:
                    return None
            
            except Exception as e:
                logger.error(f"❌ Error creating embedding: {e}")
                return None
        
        return None
    
    def load_pdf_documents(self, pdf_directory: str = "knowledge_base") -> List[Dict]:
        """
        Load and process all PDF documents in the directory with enhanced metadata.
        
        Args:
            pdf_directory: Directory containing PDF files
        
        Returns:
            List of document chunks with metadata
        """
        pdf_path = Path(pdf_directory)
        if not pdf_path.exists():
            logger.warning(f"⚠️ PDF directory {pdf_directory} does not exist")
            return []
        
        all_chunks = []
        pdf_files = list(pdf_path.glob("*.pdf"))
        
        logger.info(f"📚 Found {len(pdf_files)} PDF files to process")
        
        for pdf_file in pdf_files:
            try:
                logger.info(f"📄 Processing {pdf_file.name}...")
                
                # Get book metadata
                book_info = BOOK_METADATA.get(pdf_file.name, {
                    "title": pdf_file.stem,
                    "author": "Unknown",
                    "content_type": "therapeutic_content",
                    "topics": []
                })
                
                # Load PDF
                loader = PyPDFLoader(str(pdf_file))
                documents = loader.load()
                
                # Split documents into chunks
                chunks = self.text_splitter.split_documents(documents)
                
                for i, chunk in enumerate(chunks):
                    # Extract page number from langchain metadata
                    page_num = chunk.metadata.get("page", 0) + 1  # 0-indexed to 1-indexed
                    
                    # Create enhanced metadata
                    enhanced_metadata = {
                        "source_file": pdf_file.name,
                        "book_title": book_info["title"],
                        "author": book_info["author"],
                        "page": page_num,
                        "chunk_id": f"{pdf_file.stem}_chunk_{i}",
                        "content_type": book_info["content_type"],
                        "topics": book_info["topics"],
                        "created_at": datetime.utcnow().isoformat(),
                        "char_count": len(chunk.page_content)
                    }
                    
                    all_chunks.append({
                        "source_file": pdf_file.name,
                        "chunk_index": i,
                        "content": chunk.page_content,
                        "doc_metadata": json.dumps(enhanced_metadata)
                    })
                
                logger.info(f"✅ Created {len(chunks)} chunks from {pdf_file.name}")
            
            except Exception as e:
                logger.error(f"❌ Error processing {pdf_file.name}: {e}", exc_info=True)
                continue
        
        return all_chunks
    
    def index_documents(self, db: Session, chunks: List[Dict]) -> Tuple[int, int]:
        """
        Create embeddings and store in database.
        
        Args:
            db: Database session
            chunks: List of document chunks
        
        Returns:
            Tuple of (successful_count, failed_count)
        """
        logger.info(f"🔄 Indexing {len(chunks)} document chunks...")
        
        successful = 0
        failed = 0
        
        for i, chunk in enumerate(chunks):
            try:
                # Create embedding with retry logic
                embedding = self.create_embedding(chunk["content"])
                
                if embedding is None:
                    logger.error(f"❌ Failed to create embedding for chunk {i}")
                    failed += 1
                    continue
                
                # Store in database
                doc = KnowledgeDocument(
                    source_file=chunk["source_file"],
                    chunk_index=chunk["chunk_index"],
                    content=chunk["content"],
                    embedding=embedding,
                    doc_metadata=chunk["doc_metadata"]
                )
                db.add(doc)
                successful += 1
                
                # Commit in batches
                if (i + 1) % 10 == 0:
                    db.commit()
                    logger.info(f"✅ Indexed {i + 1}/{len(chunks)} chunks")
            
            except Exception as e:
                logger.error(f"❌ Error indexing chunk {i}: {e}")
                failed += 1
                continue
        
        db.commit()
        logger.info(f"✅ Document indexing complete! Success: {successful}, Failed: {failed}")
        return successful, failed
    
    def retrieve_relevant_context(
        self, 
        db: Session, 
        query: str, 
        k: int = 5
    ) -> List[Dict[str, str]]:
        """
        Retrieve most relevant context chunks with source metadata for citations.
        
        Args:
            db: Database session
            query: User query
            k: Number of results to retrieve
        
        Returns:
            List of dicts with 'content', 'source', 'page', 'book_title'
        """
        try:
            # Create embedding for query
            query_embedding = self.create_embedding(query)
            
            if query_embedding is None:
                logger.error("❌ Failed to create query embedding")
                return []
            
            # Search for similar documents using pgvector
            results = db.query(KnowledgeDocument).order_by(
                KnowledgeDocument.embedding.cosine_distance(query_embedding)
            ).limit(k).all()
            
            # Extract content and metadata for citations
            contexts_with_sources = []
            for doc in results:
                try:
                    metadata = json.loads(doc.doc_metadata) if doc.doc_metadata else {}
                    
                    contexts_with_sources.append({
                        "content": doc.content,
                        "source_file": doc.source_file,
                        "book_title": metadata.get("book_title", "Unknown"),
                        "author": metadata.get("author", "Unknown"),
                        "page": metadata.get("page", "N/A"),
                        "chunk_id": metadata.get("chunk_id", "unknown")
                    })
                except Exception as e:
                    logger.error(f"❌ Error parsing metadata: {e}")
                    # Fallback to basic info
                    contexts_with_sources.append({
                        "content": doc.content,
                        "source_file": doc.source_file,
                        "book_title": "Unknown",
                        "author": "Unknown",
                        "page": "N/A",
                        "chunk_id": "unknown"
                    })
            
            logger.info(f"✅ Retrieved {len(contexts_with_sources)} relevant context chunks")
            return contexts_with_sources
        
        except Exception as e:
            logger.error(f"❌ Error retrieving context: {e}", exc_info=True)
            return []
    
    def build_prompt_with_context(
        self, 
        user_message: str, 
        contexts: List[Dict[str, str]], 
        conversation_history: List[Dict] = None
    ) -> str:
        """
        Build a prompt with retrieved context and conversation history.
        Includes citations for source tracking.
        
        Args:
            user_message: Current user message
            contexts: Retrieved context chunks with metadata
            conversation_history: Previous conversation messages
        
        Returns:
            Complete prompt for GPT-4
        """
        
        # System prompt for therapeutic chatbot with citation instructions
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

IMPORTANT - Citations:
- When referencing specific concepts from the knowledge base, cite your sources
- Use format: [Book Title, p.XX] or [Source] at the end of relevant sentences
- Example: "The Four Aces framework helps build happiness through awareness. [The Four Aces, p.25]"
- Only cite when directly using information from the context provided

Safety:
- If someone mentions self-harm or suicide, provide crisis resources immediately
- Maintain professional boundaries
- Recommend professional help for serious mental health concerns

Use the provided knowledge base context to inform your responses with the Four Aces, 7Cs, and 8Ps frameworks."""
        
        # Build context section with source information
        context_section = "\n\n=== RELEVANT KNOWLEDGE BASE ===\n"
        if contexts:
            for i, ctx in enumerate(contexts, 1):
                source_info = f"[{ctx['book_title']}, p.{ctx['page']}]"
                context_section += f"\n[Context {i}] {source_info}\n{ctx['content']}\n"
        else:
            context_section += "\n(No specific context retrieved - use general knowledge)\n"
        
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


def initialize_knowledge_base() -> bool:
    """
    Initialize knowledge base by loading and indexing PDFs.
    Smart initialization that checks if embeddings already exist.
    
    Returns:
        True if successful, False otherwise
    """
    logger.info("🚀 Starting knowledge base initialization...")
    
    # Get OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key or openai_api_key == "your_openai_api_key_here":
        logger.warning("⚠️ OpenAI API key not set. Skipping knowledge base initialization.")
        return False
    
    # Initialize RAG system
    try:
        rag = TherapeuticRAG(openai_api_key)
    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG system: {e}")
        return False
    
    # Create database session
    db = SessionLocal()
    
    try:
        # Check if knowledge base already exists
        existing_docs = db.query(KnowledgeDocument).count()
        if existing_docs > 0:
            logger.info(f"✅ Knowledge base already initialized with {existing_docs} documents")
            logger.info("⏩ Skipping PDF processing (embeddings exist)")
            return True
        
        logger.info("📚 No existing embeddings found. Processing PDFs...")
        
        # Load PDF documents
        chunks = rag.load_pdf_documents()
        
        if not chunks:
            logger.warning("⚠️ No PDF documents found to index")
            return False
        
        # Index documents
        successful, failed = rag.index_documents(db, chunks)
        
        if successful > 0:
            logger.info(f"✅ Knowledge base initialization complete! Indexed {successful} chunks")
            return True
        else:
            logger.error(f"❌ Knowledge base initialization failed. No chunks indexed.")
            return False
    
    except Exception as e:
        logger.error(f"❌ Error initializing knowledge base: {e}", exc_info=True)
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

"""
Database models and configuration for PostgreSQL with pgvector.

Handles user management, conversation history, and message storage.

UPDATED: Uses aichatusers table with source_channel tracking for WhatsApp + Web.
"""

from sqlalchemy import create_engine, Column, String, Text, DateTime, Integer, Float, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from datetime import datetime
import uuid
import os
import logging
from pgvector.sqlalchemy import Vector

# Logging
logger = logging.getLogger(__name__)

# Base class for models
Base = declarative_base()

# Global references for engine and SessionLocal (set by server.py)
engine = None
SessionLocal = None

def set_engine_and_session(engine_instance, session_factory):
    """Set engine and session factory from server.py"""
    global engine, SessionLocal
    engine = engine_instance
    SessionLocal = session_factory

# ======================================================================
# USER MODEL - WITH source_channel FOR WHATSAPP + WEB TRACKING
# ======================================================================

class aichatusers(Base):
    """User model for tracking WhatsApp and Web users with source channel tracking."""
    
    __tablename__ = "aichatusers"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    whatsapp_number = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=True)
    first_interaction = Column(DateTime, default=datetime.utcnow)
    last_interaction = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    total_messages = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    crisis_flag = Column(Boolean, default=False)  # Flag for crisis intervention
    created_at = Column(DateTime, default=datetime.utcnow)
    source_channel = Column(String(50), default='whatsapp', nullable=False)  # NEW: Track source (whatsapp or web)
    
    def __repr__(self):
        return f"<aichatusers(id={self.id}, whatsapp_number={self.whatsapp_number}, source_channel={self.source_channel})>"

# Backward compatibility alias
User = aichatusers

# ======================================================================
# CONVERSATION MODEL
# ======================================================================

class Conversation(Base):
    """Conversation model for tracking chat sessions."""
    __tablename__ = "conversations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    started_at = Column(DateTime, default=datetime.utcnow)
    last_message_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    message_count = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    
    def __repr__(self):
        return f"<Conversation {self.id} for User {self.user_id}>"

# ======================================================================
# MESSAGE MODEL
# ======================================================================

class Message(Base):
    """Message model for storing chat messages with context."""
    __tablename__ = "messages"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    role = Column(String(20), nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    embedding = Column(Vector(1536))  # OpenAI text-embedding-3-large dimension
    sentiment_score = Column(Float, nullable=True)
    contains_crisis_keywords = Column(Boolean, default=False)
    
    def __repr__(self):
        return f"<Message {self.id} from {self.role}>"

# ======================================================================
# KNOWLEDGE DOCUMENT MODEL
# ======================================================================

class KnowledgeDocument(Base):
    """Knowledge base document model for storing PDF chunks with embeddings."""
    __tablename__ = "knowledge_documents"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_file = Column(String(255), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(1536))  # OpenAI text-embedding-3-large dimension
    doc_metadata = Column(Text, nullable=True)  # JSON string with additional info
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<KnowledgeDocument {self.source_file} chunk {self.chunk_index}>"

# ======================================================================
# DATABASE INITIALIZATION
# ======================================================================

def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully!")

def get_db():
    """Dependency for getting database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ======================================================================
# USER MANAGEMENT WITH source_channel TRACKING
# ======================================================================

def get_or_create_user(db, whatsapp_number: str):
    """
    Get existing user or create new one with source_channel tracking.
    
    Auto-detects source_channel:
    - If whatsapp_number starts with 'web:' → source_channel = 'web'
    - Otherwise → source_channel = 'whatsapp'
    
    Args:
        db: SQLAlchemy session
        whatsapp_number: User identifier (phone or web:uuid format)
    
    Returns:
        aichatusers object (new or existing)
    """
    
    # Determine source_channel based on whatsapp_number format
    source_channel = 'web' if whatsapp_number.startswith('web:') else 'whatsapp'
    
    # Check if user exists
    user = db.query(aichatusers).filter(aichatusers.whatsapp_number == whatsapp_number).first()
    
    if not user:
        # Create new user with total_messages = 1 (for this first interaction)
        user = aichatusers(
            whatsapp_number=whatsapp_number,
            source_channel=source_channel,
            total_messages=1  # FIX: Count the first message
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        logger.info(f"✓ New user created: {whatsapp_number} (source: {source_channel})")
    else:
        # Update existing user
        user.last_interaction = datetime.utcnow()
        user.total_messages += 1

        # Update source_channel if not set or if changed
        if not user.source_channel or user.source_channel == 'whatsapp':
            user.source_channel = source_channel

        db.commit()
        logger.info(f"✓ User updated: {whatsapp_number} (source: {source_channel})")
    
    return user

# ======================================================================
# CONVERSATION MANAGEMENT
# ======================================================================

def get_active_conversation(db, user_id: uuid.UUID):
    """Get or create active conversation for user."""
    conversation = db.query(Conversation).filter(
        Conversation.user_id == user_id,
        Conversation.is_active == True
    ).first()
    
    if not conversation:
        conversation = Conversation(user_id=user_id)
        db.add(conversation)
        db.commit()
        db.refresh(conversation)
        logger.info(f"✓ New conversation created for user {user_id}")
    
    return conversation

# ======================================================================
# MESSAGE MANAGEMENT
# ======================================================================

def save_message(db, conversation_id: uuid.UUID, user_id: uuid.UUID, 
                role: str, content: str, embedding=None, contains_crisis=False):
    """
    Save a message to the database.
    
    Args:
        db: SQLAlchemy session
        conversation_id: UUID of conversation
        user_id: UUID of user
        role: 'user' or 'assistant'
        content: Message text
        embedding: Vector embedding (optional)
        contains_crisis: Crisis flag (optional)
    
    Returns:
        Message object
    """
    message = Message(
        conversation_id=conversation_id,
        user_id=user_id,
        role=role,
        content=content,
        embedding=embedding,
        contains_crisis_keywords=contains_crisis
    )
    db.add(message)
    
    # Update conversation
    conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if conversation:
        conversation.message_count += 1
        conversation.last_message_at = datetime.utcnow()
    
    db.commit()
    db.refresh(message)
    logger.info(f"✓ Message saved to conversation {conversation_id}")
    return message

def get_conversation_history(db, conversation_id: uuid.UUID, limit: int = 10):
    """
    Get recent conversation history.
    
    Args:
        db: SQLAlchemy session
        conversation_id: UUID of conversation
        limit: Number of messages to retrieve
    
    Returns:
        List of Message objects in chronological order
    """
    messages = db.query(Message).filter(
        Message.conversation_id == conversation_id
    ).order_by(Message.timestamp.desc()).limit(limit).all()
    
    return list(reversed(messages))  # Returns in chronological order

# ======================================================================
# VECTOR SEARCH (RAG)
# ======================================================================

def search_similar_messages(db, embedding, limit: int = 5):
    """
    Search for similar past messages using vector similarity.
    
    Args:
        db: SQLAlchemy session
        embedding: Vector embedding to search for
        limit: Number of similar messages to return
    
    Returns:
        List of similar Message objects
    """
    # This uses pgvector's cosine distance operator
    similar_messages = db.query(Message).order_by(
        Message.embedding.cosine_distance(embedding)
    ).limit(limit).all()
    
    logger.info(f"✓ Found {len(similar_messages)} similar messages")
    return similar_messages

# ======================================================================
# INITIALIZATION
# ======================================================================

if __name__ == "__main__":
    logger.info("Initializing database...")
    init_db()
    logger.info("✓ Database ready!")


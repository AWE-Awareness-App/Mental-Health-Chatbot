"""
Database models and configuration for PostgreSQL with pgvector.
Handles user management, conversation history, and message storage.
"""

from sqlalchemy import create_engine, Column, String, Text, DateTime, Integer, Float, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from datetime import datetime
import uuid
import os
from pgvector.sqlalchemy import Vector

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://chatbot_user:chatbot_pass@localhost:5432/therapy_chatbot")

# Create engine
engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=10, max_overflow=20)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


class User(Base):
    """User model for tracking WhatsApp users."""
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    whatsapp_number = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=True)
    first_interaction = Column(DateTime, default=datetime.utcnow)
    last_interaction = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    total_messages = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    crisis_flag = Column(Boolean, default=False)  # Flag for crisis intervention
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<User {self.whatsapp_number}>"


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


# Database initialization functions
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


# Utility functions for database operations
def get_or_create_user(db, whatsapp_number: str):
    """Get existing user or create new one."""
    user = db.query(User).filter(User.whatsapp_number == whatsapp_number).first()
    if not user:
        user = User(whatsapp_number=whatsapp_number)
        db.add(user)
        db.commit()
        db.refresh(user)
    else:
        user.last_interaction = datetime.utcnow()
        user.total_messages += 1
        db.commit()
    return user


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
    
    return conversation


def save_message(db, conversation_id: uuid.UUID, user_id: uuid.UUID, 
                role: str, content: str, embedding=None, contains_crisis=False):
    """Save a message to the database."""
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
    return message


def get_conversation_history(db, conversation_id: uuid.UUID, limit: int = 10):
    """Get recent conversation history."""
    messages = db.query(Message).filter(
        Message.conversation_id == conversation_id
    ).order_by(Message.timestamp.desc()).limit(limit).all()
    
    return list(reversed(messages))  # Return in chronological order


def search_similar_messages(db, embedding, limit: int = 5):
    """Search for similar past messages using vector similarity."""
    # This uses pgvector's cosine distance operator
    similar_messages = db.query(Message).order_by(
        Message.embedding.cosine_distance(embedding)
    ).limit(limit).all()
    
    return similar_messages


if __name__ == "__main__":
    print("Initializing database...")
    init_db()
    print("Database ready!")

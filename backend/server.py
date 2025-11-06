"""
AWE Mental Health Chatbot - FastAPI Server
Handles WhatsApp messaging with therapeutic AI responses
"""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

# Database setup with Azure AD
from database_aad import get_database_engine
from database import Base, set_engine_and_session
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text

# Import chatbot
from chatbot import TherapeuticChatbot
from rag_system_v2 import TherapeuticRAG

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ===== DATABASE SETUP WITH AZURE AD =====
# Create engine with Azure AD authentication
engine = get_database_engine()

# Create SessionLocal factory
session_factory = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False
)

# CRITICAL: Tell database.py to use our Azure AD engine!
set_engine_and_session(engine, session_factory)

# Module-level SessionLocal for backward compatibility
SessionLocal = session_factory

# ===== APPLICATION STARTUP/SHUTDOWN =====
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager for startup and shutdown events
    """
    # ===== STARTUP =====
    logger.info("🚀 Starting chatbot initialization...")
    
    try:
        # Test database connection with Azure AD
        logger.info("🔐 Testing Azure AD database connection...")
        with SessionLocal() as db:
            db.execute(text("SELECT 1"))
        logger.info("✓ Database connection verified")
    except Exception as e:
        logger.error(f"✗ Database initialization failed: {e}")
        raise
    
    # Initialize RAG system
    try:
        logger.info("📚 Initializing knowledge base (RAG system)...")
        rag_system = TherapeuticRAG()
        logger.info("✓ Knowledge base initialized")
    except Exception as e:
        logger.warning(f"⚠️ RAG system initialization failed: {e}")
        rag_system = None
    
    # Initialize chatbot
    try:
        logger.info("🤖 Initializing therapeutic chatbot...")
        chatbot = TherapeuticChatbot()
        logger.info("✓ Chatbot initialized")
        app.state.chatbot = chatbot
        app.state.rag_system = rag_system
    except Exception as e:
        logger.error(f"✗ Chatbot initialization failed: {e}")
        raise
    
    logger.info("""
    
╔══════════════════════════════════════════════════════════════╗
║       WhatsApp Therapeutic Chatbot - Production MVP         ║
║              Digital Wellness & Mental Health                ║
╚══════════════════════════════════════════════════════════════╝
    
    """)
    logger.info("🎉 Chatbot startup complete and ready to serve!")
    
    yield
    
    # ===== SHUTDOWN =====
    logger.info("👋 Shutting down chatbot gracefully...")
    try:
        engine.dispose()
        logger.info("✓ Database connections closed")
    except Exception as e:
        logger.error(f"✗ Error during shutdown: {e}")
    logger.info("👋 Shutdown complete")

# ===== FASTAPI APP =====
app = FastAPI(
    title="AWE Mental Health Chatbot",
    description="Therapeutic chatbot for digital wellness via WhatsApp",
    version="1.0.0",
    lifespan=lifespan
)

# ===== CORS CONFIGURATION =====
cors_origins = os.getenv("CORS_ORIGINS", "").split(",") if os.getenv("CORS_ORIGINS") else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== ROUTES =====

@app.get("/")
async def root():
    """Root endpoint - basic health check"""
    return {
        "status": "online",
        "service": "WhatsApp Therapeutic Chatbot",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/health")
async def health():
    """Health check endpoint"""
    try:
        # Verify database connection
        with SessionLocal() as db:
            db.execute(text("SELECT 1"))
        
        return {
            "status": "healthy",
            "service": "WhatsApp Therapeutic Chatbot",
            "version": "1.0.0",
            "database": "connected",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Database connection failed")

@app.get("/api/status")
async def status():
    """Service status endpoint"""
    return {
        "status": "operational",
        "service": "AWE Mental Health Chatbot",
        "version": "1.0.0",
        "environment": os.getenv("ENVIRONMENT", "production"),
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/api/whatsapp")
async def whatsapp_webhook(request: Request):
    """
    WhatsApp webhook endpoint
    Receives messages from Twilio and routes to chatbot
    """
    try:
        form_data = await request.form()
        incoming_message = form_data.get("Body", "")
        phone_number = form_data.get("From", "")
        
        logger.info(f"📱 Received message from {phone_number}: {incoming_message}")
        
        if not incoming_message:
            logger.warning("Empty message received")
            return {"status": "processed"}
        
        # Process message through chatbot
        db = SessionLocal()
        try:
            chatbot = app.state.chatbot
            response = chatbot.generate_response(
                phone_number=phone_number,
                user_message=incoming_message,
                db=db
            )
            
            logger.info(f"✓ Generated response: {response[:100]}...")
            return {"status": "processed", "message": response}
            
        except Exception as e:
            logger.error(f"✗ Error processing message: {e}")
            return {
                "status": "error",
                "message": "I apologize, but I'm having trouble processing your message right now. This is a temporary technical issue on my end.\n\nPlease try again in a moment. If you're experiencing a mental health crisis, please contact:\n- 988 Suicide & Crisis Lifeline\n- Crisis Text Line: Text HOME to 741741\n\nI'm here to help with digital wellness when you're ready to try again."
            }
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"✗ Webhook error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/test-message")
async def test_message(message: dict):
    """Test endpoint for sending messages directly"""
    try:
        user_message = message.get("message", "")
        phone_number = message.get("phone", "test")
        
        db = SessionLocal()
        try:
            chatbot = app.state.chatbot
            response = chatbot.generate_response(
                phone_number=phone_number,
                user_message=user_message,
                db=db
            )
            return {"response": response}
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Test message error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== ERROR HANDLERS =====
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Internal server error",
            "detail": str(exc)
        }
    )

# ===== RUN SERVER =====
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8001)),
        log_level="info"
    )
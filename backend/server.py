"""
AWE Mental Health Chatbot - FastAPI Server
Handles BOTH WhatsApp (Twilio) AND Web Chat messaging with therapeutic AI responses
Multi-Channel Support: WhatsApp + Web Frontend with Conversation Tracking
"""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import uuid

# Twilio for WhatsApp responses
from twilio.rest import Client

# Database setup
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

# ===== DATABASE SETUP =====
engine = get_database_engine()
session_factory = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False
)

set_engine_and_session(engine, session_factory)
SessionLocal = session_factory

# ===== TWILIO SETUP =====
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER", "whatsapp:+14155238886")

# Initialize Twilio client
twilio_client = None
if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
    twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    logger.info(f"✓ Twilio client initialized with number: {TWILIO_WHATSAPP_NUMBER}")
else:
    logger.warning("⚠️ Twilio credentials not set - WhatsApp responses will not be sent")

# ===== APPLICATION STARTUP/SHUTDOWN =====
@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager for startup and shutdown events"""
    
    # ===== STARTUP =====
    logger.info("🚀 Starting chatbot initialization...")
    
    try:
        logger.info("🔐 Testing database connection...")
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
    
    # Initialize chatbot with OpenAI API key
    try:
        logger.info("🤖 Initializing therapeutic chatbot...")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not openai_api_key:
            logger.error("✗ OPENAI_API_KEY environment variable not set!")
            raise ValueError("OPENAI_API_KEY is required")
        
        chatbot = TherapeuticChatbot(openai_api_key=openai_api_key)
        logger.info("✓ Chatbot initialized successfully")
        
        app.state.chatbot = chatbot
        app.state.rag_system = rag_system
    except Exception as e:
        logger.error(f"✗ Chatbot initialization failed: {e}")
        raise
    
    logger.info("""
    ╔══════════════════════════════════════════════════════════════╗
    ║   AWE Therapeutic Chatbot - Multi-Channel Production MVP   ║
    ║         WhatsApp + Web Chat | Digital Wellness              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    logger.info("🎉 Chatbot startup complete and ready to serve!")
    logger.info("📱 WhatsApp channel: ACTIVE")
    logger.info("💻 Web chat channel: ACTIVE")
    
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
    title="AWE Mental Health Chatbot - Multi-Channel",
    description="Therapeutic chatbot for digital wellness via WhatsApp and Web",
    version="1.0.0",
    lifespan=lifespan
)

# ===== CORS CONFIGURATION =====
# Allow web frontend to connect
cors_origins = [
    "http://localhost:3000",              # Local dev
    "http://localhost:3001",
    "https://awedigitalwellness.com",     # Production
    "https://www.awedigitalwellness.com",
    "https://*.vercel.app",               # Vercel previews
]

# Also check environment variable
env_origins = os.getenv("CORS_ORIGINS", "").split(",") if os.getenv("CORS_ORIGINS") else []
if env_origins:
    cors_origins.extend(env_origins)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins if cors_origins else ["*"],
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
        "service": "AWE Therapeutic Chatbot - Multi-Channel",
        "channels": ["whatsapp", "web"],
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/health")
async def health():
    """Health check endpoint"""
    try:
        with SessionLocal() as db:
            db.execute(text("SELECT 1"))
        
        return {
            "status": "healthy",
            "service": "AWE Therapeutic Chatbot",
            "channels": {
                "whatsapp": "active" if twilio_client else "inactive",
                "web": "active"
            },
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
        "channels": {
            "whatsapp": {
                "status": "active" if twilio_client else "inactive",
                "number": TWILIO_WHATSAPP_NUMBER if twilio_client else None
            },
            "web": {
                "status": "active",
                "endpoints": ["/api/webChat", "/api/awe-chat"]
            }
        },
        "version": "1.0.0",
        "environment": os.getenv("ENVIRONMENT", "production"),
        "timestamp": datetime.utcnow().isoformat()
    }


# ═══════════════════════════════════════════════════════════════════
# CHANNEL 1: WhatsApp (Twilio)
# ═══════════════════════════════════════════════════════════════════

@app.post("/api/whatsapp")
async def whatsapp_webhook(request: Request):
    """
    WhatsApp webhook endpoint
    Receives messages from Twilio and sends responses back
    """
    try:
        form_data = await request.form()
        incoming_message = form_data.get("Body", "").strip()
        whatsapp_number = form_data.get("From", "")
        
        logger.info(f"📱 Received WhatsApp message from {whatsapp_number}: {incoming_message}")
        
        if not incoming_message:
            logger.warning("Empty message received")
            return {"status": "processed"}
        
        # Process message through chatbot
        db = SessionLocal()
        try:
            chatbot = app.state.chatbot
            
            # Generate response from chatbot
            response_dict = chatbot.generate_response(
                db=db,
                whatsapp_number=whatsapp_number,
                user_message=incoming_message
            )
            
            response_text = response_dict.get("response", "")
            logger.info(f"✓ Generated WhatsApp response: {response_text[:100]}...")
            
            # Send response back via Twilio WhatsApp
            if twilio_client:
                try:
                    message = twilio_client.messages.create(
                        from_=TWILIO_WHATSAPP_NUMBER,
                        body=response_text,
                        to=whatsapp_number
                    )
                    
                    logger.info(f"✓ WhatsApp message sent to {whatsapp_number} (SID: {message.sid})")
                    
                    return {
                        "status": "sent",
                        "message_sid": message.sid,
                        "response": response_text
                    }
                except Exception as e:
                    logger.error(f"✗ Failed to send message via Twilio: {e}")
                    return {
                        "status": "error",
                        "message": "Failed to send response back",
                        "detail": str(e)
                    }
            else:
                logger.warning("⚠️ Twilio client not configured - not sending response")
                return {
                    "status": "processed",
                    "message": response_text,
                    "note": "Twilio not configured"
                }
        
        except Exception as e:
            logger.error(f"✗ Error processing WhatsApp message: {e}")
            
            # Send error message
            if twilio_client:
                try:
                    twilio_client.messages.create(
                        from_=TWILIO_WHATSAPP_NUMBER,
                        body="I apologize, but I'm having trouble processing your message right now. This is a temporary technical issue on my end.\n\nPlease try again in a moment. If you're experiencing a mental health crisis, please contact:\n- 988 Suicide & Crisis Lifeline\n- Crisis Text Line: Text HOME to 741741",
                        to=whatsapp_number
                    )
                except Exception as send_error:
                    logger.error(f"✗ Failed to send error message: {send_error}")
            
            return {"status": "error", "detail": str(e)}
        finally:
            db.close()
    
    except Exception as e:
        logger.error(f"✗ WhatsApp webhook error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════
# CHANNEL 2: Web Chat with Conversation Tracking - PRIMARY ENDPOINT
# ═══════════════════════════════════════════════════════════════════

@app.post("/api/webChat")
async def web_chat_tracked(request: Request):
    """
    Enhanced web chat endpoint with conversation tracking
    Matches frontend developer's requested format
    
    Request format:
    {
        "conversation_id": "uuid-or-null",
        "message_index": 0,
        "content": "user message text"
    }
    
    Response format:
    {
        "conversation_id": "uuid",
        "message_index": 1,
        "content": "bot response text",
        "is_crisis": boolean (optional)
    }
    """
    try:
        body = await request.json()
        user_message = body.get("content", "").strip()
        conversation_id = body.get("conversation_id")
        message_index = body.get("message_index", 0)
        
        logger.info(f"💻 Web chat - Conv: {conversation_id}, Msg#{message_index}, Content: {user_message[:50]}...")
        
        if not user_message:
            return JSONResponse({
                "conversation_id": conversation_id,
                "message_index": message_index + 1,
                "content": "Please enter a message.",
                "error": True
            }, status_code=400)
        
        # Process through chatbot
        db = SessionLocal()
        try:
            chatbot = app.state.chatbot
            
            # Generate or use existing conversation ID
            if not conversation_id:
                conversation_id = str(uuid.uuid4())
                logger.info(f"✓ Created new conversation: {conversation_id}")
            
            # Use conversation_id as user identifier for web users
            web_user_id = f"web:{conversation_id}"
            
            # Generate response
            response_dict = chatbot.generate_response(
                db=db,
                whatsapp_number=web_user_id,
                user_message=user_message
            )
            
            response_text = response_dict.get("response", "")
            is_crisis = response_dict.get("is_crisis", False)
            
            logger.info(f"✓ Web chat response - Conv: {conversation_id}, Msg#{message_index + 1}, Length: {len(response_text)} chars")
            
            # Return in requested format
            return JSONResponse({
                "conversation_id": conversation_id,
                "message_index": message_index + 1,
                "content": response_text,
                "is_crisis": is_crisis
            })
        
        except Exception as e:
            logger.error(f"✗ Error processing web chat: {e}", exc_info=True)
            return JSONResponse({
                "conversation_id": conversation_id,
                "message_index": message_index + 1,
                "content": "I apologize, but I'm having trouble processing your message right now. Please try again in a moment.",
                "error": True
            }, status_code=500)
        finally:
            db.close()
    
    except Exception as e:
        logger.error(f"✗ Web chat endpoint error: {e}", exc_info=True)
        return JSONResponse({
            "conversation_id": None,
            "message_index": 0,
            "content": "Sorry, something went wrong. Please try again.",
            "error": True
        }, status_code=500)


# ═══════════════════════════════════════════════════════════════════
# CHANNEL 2B: Alternative Web Chat Endpoint (Backward Compatible)
# ═══════════════════════════════════════════════════════════════════

@app.post("/api/awe-chat")
async def awe_chat(request: Request):
    """
    Alternative web chat endpoint (backward compatible)
    Simpler format without conversation tracking
    
    Request format:
    {
        "message": "user message text",
        "history": [{"role": "user|assistant", "content": "..."}]
    }
    
    Response format:
    {
        "reply": "bot response text",
        "is_crisis": boolean,
        "user_id": "web-timestamp"
    }
    """
    try:
        body = await request.json()
        user_message = body.get("message", "").strip()
        message_history = body.get("history", [])
        
        # Generate unique web user ID
        user_id = f"web-{datetime.utcnow().timestamp()}"
        
        logger.info(f"💻 Web chat (simple) - Message: {user_message[:50]}...")
        
        if not user_message:
            return JSONResponse({
                "reply": "Please enter a message.",
                "error": True
            }, status_code=400)
        
        # Process through chatbot
        db = SessionLocal()
        try:
            chatbot = app.state.chatbot
            
            response_dict = chatbot.generate_response(
                db=db,
                whatsapp_number=f"web:{user_id}",
                user_message=user_message
            )
            
            response_text = response_dict.get("response", "")
            is_crisis = response_dict.get("is_crisis", False)
            
            logger.info(f"✓ Web chat (simple) response - Length: {len(response_text)} chars")
            
            return JSONResponse({
                "reply": response_text,
                "is_crisis": is_crisis,
                "user_id": user_id
            })
        
        except Exception as e:
            logger.error(f"✗ Error processing web chat: {e}", exc_info=True)
            return JSONResponse({
                "reply": "I apologize, but I'm having trouble processing your message right now. Please try again in a moment.",
                "error": True
            }, status_code=500)
        finally:
            db.close()
    
    except Exception as e:
        logger.error(f"✗ Web chat endpoint error: {e}", exc_info=True)
        return JSONResponse({
            "reply": "Sorry, something went wrong. Please try again.",
            "error": True
        }, status_code=500)


# ═══════════════════════════════════════════════════════════════════
# TESTING & DEBUGGING ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@app.post("/api/test-message")
async def test_message(message: dict):
    """Test endpoint for sending messages directly"""
    try:
        user_message = message.get("message", "")
        whatsapp_number = message.get("phone", "test")
        
        db = SessionLocal()
        try:
            chatbot = app.state.chatbot
            
            response_dict = chatbot.generate_response(
                db=db,
                whatsapp_number=whatsapp_number,
                user_message=user_message
            )
            
            return {"response": response_dict.get("response", "")}
        finally:
            db.close()
    
    except Exception as e:
        logger.error(f"Test message error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════
# ERROR HANDLERS
# ═══════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════
# RUN SERVER
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8001)),
        log_level="info"
    )

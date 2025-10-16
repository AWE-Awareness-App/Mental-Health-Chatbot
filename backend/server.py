"""
FastAPI server for WhatsApp Therapeutic Chatbot.
Handles Twilio WhatsApp webhook integration and chatbot responses.
"""

from fastapi import FastAPI, APIRouter, Request, HTTPException, Depends
from fastapi.responses import PlainTextResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from twilio.twiml.messaging_response import MessagingResponse
from twilio.request_validator import RequestValidator
from sqlalchemy.orm import Session
from pathlib import Path
import os
import logging
from contextlib import asynccontextmanager

# Import local modules
from database import init_db, get_db
from chatbot import get_chatbot
from rag_system import initialize_knowledge_base

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/chatbot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# Twilio configuration
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER")

# Create logs directory
os.makedirs('logs', exist_ok=True)


# Lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    # Startup
    logger.info("Starting WhatsApp Therapeutic Chatbot...")
    
    try:
        # Initialize database
        logger.info("Initializing database...")
        init_db()
        
        # Initialize knowledge base
        logger.info("Initializing knowledge base...")
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key and openai_key != "your_openai_api_key_here":
            initialize_knowledge_base()
        else:
            logger.warning("OpenAI API key not configured. Knowledge base not initialized.")
        
        logger.info("Chatbot ready!")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down chatbot...")


# Create FastAPI app
app = FastAPI(
    title="WhatsApp Therapeutic Chatbot",
    description="Digital wellness chatbot for managing screen time and technology addiction",
    version="1.0.0",
    lifespan=lifespan
)

# Create API router
api_router = APIRouter(prefix="/api")


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@api_router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "WhatsApp Therapeutic Chatbot",
        "version": "1.0.0"
    }


@api_router.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "message": "WhatsApp Therapeutic Chatbot API",
        "status": "running",
        "endpoints": {
            "health": "/api/health",
            "whatsapp_webhook": "/api/whatsapp (POST)",
            "status": "/api/status"
        }
    }


@api_router.get("/status")
async def status(db: Session = Depends(get_db)):
    """Get service status and statistics."""
    try:
        from database import User, Message, KnowledgeDocument
        
        total_users = db.query(User).count()
        total_messages = db.query(Message).count()
        knowledge_docs = db.query(KnowledgeDocument).count()
        
        return {
            "status": "operational",
            "database": "connected",
            "statistics": {
                "total_users": total_users,
                "total_messages": total_messages,
                "knowledge_base_documents": knowledge_docs
            },
            "configuration": {
                "twilio_configured": bool(TWILIO_ACCOUNT_SID and TWILIO_ACCOUNT_SID != "your_twilio_account_sid_here"),
                "openai_configured": bool(os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_API_KEY") != "your_openai_api_key_here")
            }
        }
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


@api_router.post("/whatsapp")
async def whatsapp_webhook(request: Request, db: Session = Depends(get_db)):
    """
    Twilio WhatsApp webhook endpoint.
    Receives messages from WhatsApp and sends therapeutic responses.
    """
    try:
        # Parse form data from Twilio
        form_data = await request.form()
        
        # Extract message details
        from_number = form_data.get("From", "")  # Format: whatsapp:+1234567890
        message_body = form_data.get("Body", "")
        
        logger.info(f"Received message from {from_number}: {message_body}")
        
        # Validate Twilio request (optional but recommended for production)
        if TWILIO_AUTH_TOKEN and TWILIO_AUTH_TOKEN != "your_twilio_auth_token_here":
            validator = RequestValidator(TWILIO_AUTH_TOKEN)
            url = str(request.url)
            signature = request.headers.get("X-Twilio-Signature", "")
            
            # Note: In production, enable this validation
            # if not validator.validate(url, dict(form_data), signature):
            #     logger.warning(f"Invalid Twilio signature from {from_number}")
            #     raise HTTPException(status_code=403, detail="Invalid signature")
        
        # Check if message is empty
        if not message_body.strip():
            response = MessagingResponse()
            response.message("I received an empty message. How can I help you today with digital wellness?")
            return PlainTextResponse(str(response), media_type="application/xml")
        
        # Get chatbot instance
        try:
            chatbot = get_chatbot()
        except ValueError as e:
            logger.error(f"Chatbot not configured: {e}")
            response = MessagingResponse()
            response.message("The chatbot is currently being configured. Please try again later.")
            return PlainTextResponse(str(response), media_type="application/xml")
        
        # Generate therapeutic response
        result = chatbot.generate_response(db, from_number, message_body)
        bot_response = result["response"]
        
        # Format for WhatsApp
        formatted_response = chatbot.format_whatsapp_message(bot_response)
        
        # Create Twilio response
        twiml_response = MessagingResponse()
        twiml_response.message(formatted_response)
        
        logger.info(f"Sent response to {from_number}")
        
        return PlainTextResponse(str(twiml_response), media_type="application/xml")
    
    except Exception as e:
        logger.error(f"Error processing WhatsApp message: {e}", exc_info=True)
        
        # Send error response to user
        response = MessagingResponse()
        response.message(
            "I apologize, but I encountered an error. Please try again. "
            "If you need immediate help, contact 988 (Suicide & Crisis Lifeline)."
        )
        return PlainTextResponse(str(response), media_type="application/xml")


@api_router.post("/test-message")
async def test_message(
    message: str,
    whatsapp_number: str = "whatsapp:+1234567890",
    db: Session = Depends(get_db)
):
    """
    Test endpoint for generating responses without Twilio.
    Useful for development and testing.
    """
    try:
        chatbot = get_chatbot()
        result = chatbot.generate_response(db, whatsapp_number, message)
        
        return {
            "success": True,
            "user_message": message,
            "bot_response": result["response"],
            "is_crisis": result["is_crisis"],
            "user_id": result["user_id"]
        }
    except ValueError as e:
        raise HTTPException(status_code=503, detail=f"Chatbot not configured: {str(e)}")
    except Exception as e:
        logger.error(f"Error in test message: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Include API router
app.include_router(api_router)


# Root endpoint (outside API prefix)
@app.get("/")
async def root_redirect():
    """Redirect to API root."""
    return {
        "message": "WhatsApp Therapeutic Chatbot",
        "api_docs": "/docs",
        "api_base": "/api"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8001, reload=True)

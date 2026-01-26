"""
AWE Mental Health Chatbot - FastAPI Server


Handles BOTH WhatsApp (Twilio) AND Web Chat messaging
with therapeutic AI responses


Multi-Channel Support: WhatsApp + Web Frontend
with Conversation Tracking
"""


import os
import logging
import uuid
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text, extract, func


# Twilio for WhatsApp
from twilio.rest import Client
from twilio.request_validator import RequestValidator


# Database setup
from sqlalchemy.orm import sessionmaker
from database_aad import get_database_engine
from database import Base, set_engine_and_session


# Chatbot + RAG
from chatbot import TherapeuticChatbot
from rag_system_v2 import TherapeuticRAG


# ======================================================================
# LOGGING CONFIGURATION
# ======================================================================


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)


logger = logging.getLogger(__name__)


# ======================================================================
# DATABASE SETUP
# ======================================================================


engine = get_database_engine()


session_factory = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False
)


set_engine_and_session(engine, session_factory)


SessionLocal = session_factory


# ======================================================================
# TWILIO SETUP
# ======================================================================


TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER")


twilio_client = None
twilio_validator = None


if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
    twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    twilio_validator = RequestValidator(TWILIO_AUTH_TOKEN)
    logger.info(f"✓ Twilio client initialized with number: {TWILIO_WHATSAPP_NUMBER}")
else:
    logger.warning("⚠️ Twilio credentials not set - WhatsApp responses will not be sent")


# ======================================================================
# VOICE INTEGRATION IMPORTS
# ======================================================================
try:
    from routes.voice_routes import router as voice_router
    VOICE_ROUTES_AVAILABLE = True
except ImportError as e:
    VOICE_ROUTES_AVAILABLE = False
    logging.warning(f"Voice routes not available: {e}")


# ======================================================================
# APPLICATION LIFESPAN: STARTUP & SHUTDOWN
# ======================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI startup and shutdown lifecycle"""
    
    # ---------- STARTUP ----------
    logger.info("🚀 Starting chatbot initialization...")
    
    # Database test and auto-create tables
    try:
        logger.info("🔐 Testing database connection...")
        with SessionLocal() as db:
            db.execute(text("SELECT 1"))
        logger.info("✓ Database connection verified")

        # Auto-create any missing tables (including new voice tables)
        from database import Base
        Base.metadata.create_all(bind=engine)
        logger.info("✓ Database tables synchronized")
    except Exception as e:
        logger.error(f"✗ Database initialization failed: {e}")
        raise
    
    # RAG Initialization
    rag_system = None
    try:
        logger.info("📚 Initializing knowledge base (RAG system)...")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            logger.warning("⚠️ OPENAI_API_KEY not set - RAG system will not be initialized")
        else:
            rag_system = TherapeuticRAG(openai_api_key=openai_api_key)
            logger.info("✓ Knowledge base initialized")
    except Exception as e:
        logger.warning(f"⚠️ RAG system initialization failed: {e}")
        rag_system = None
    
    # Chatbot Initialization
    try:
        logger.info("🤖 Initializing therapeutic chatbot...")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            logger.error("✗ OPENAI_API_KEY not set!")
            raise ValueError("OPENAI_API_KEY is required")
        
        chatbot = TherapeuticChatbot(
            openai_api_key=openai_api_key,
            rag_system=rag_system
        )
        
        app.state.chatbot = chatbot
        app.state.rag_system = rag_system
        logger.info("✓ Chatbot initialized successfully")

        # Initialize voice handler for Phase 3B integration
        try:
            from services.voice_chatbot_handler import VoiceChatbotHandler
            from routes.voice_routes import set_voice_handler
            voice_handler = VoiceChatbotHandler(chatbot)
            set_voice_handler(voice_handler)
            app.state.voice_handler = voice_handler
            logger.info("✓ Voice handler initialized successfully")
        except Exception as voice_err:
            logger.warning(f"⚠️ Voice handler initialization failed: {voice_err}")

    except Exception as e:
        logger.error(f"✗ Chatbot initialization failed: {e}")
        raise
    
    # Startup banner
    logger.info("""
╔══════════════════════════════════════════════════════════════╗
║ AWE Therapeutic Chatbot - Multi-Channel Production MVP ║
║ WhatsApp + Web Chat | Digital Wellness Support ║
╚══════════════════════════════════════════════════════════════╝
""")
    logger.info("🎉 Chatbot startup complete and ready to serve!")
    logger.info("📱 WhatsApp channel: ACTIVE")
    logger.info("💻 Web chat channel: ACTIVE")
    
    yield
    
    # ---------- SHUTDOWN ----------
    logger.info("👋 Shutting down chatbot gracefully...")
    try:
        engine.dispose()
        logger.info("✓ Database connections closed")
    except Exception as e:
        logger.error(f"✗ Shutdown error: {e}")
    
    logger.info("👋 Shutdown complete")


# ======================================================================
# FASTAPI INSTANCE
# ======================================================================


app = FastAPI(
    title="AWE Mental Health Chatbot - Multi-Channel",
    description="Therapeutic chatbot for digital wellness via WhatsApp and Web",
    version="1.0.0",
    lifespan=lifespan
)


# ======================================================================
# CORS CONFIGURATION
# ======================================================================


cors_origins = [
    "http://localhost:3000",
    "http://localhost:3001",
    "https://awedigitalwellness.com",
    "https://www.awedigitalwellness.com",
    "https://*.vercel.app",
]


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


# ======================================================================
# BASIC ROUTES
# ======================================================================


@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "AWE Therapeutic Chatbot - Multi-Channel",
        "channels": ["whatsapp", "web"],
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/api/health")
async def health():
    try:
        with SessionLocal() as db:
            db.execute(text("SELECT 1"))
        return {
            "status": "healthy",
            "service": "AWE Therapeutic Chatbot",
            "channels": {
                "whatsapp": "active" if twilio_client else "inactive",
                "web": "active",
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
    return {
        "status": "operational",
        "service": "AWE Mental Health Chatbot",
        "channels": {
            "whatsapp": {
                "status": "active" if twilio_client else "inactive",
                "number": TWILIO_WHATSAPP_NUMBER if twilio_client else None,
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


# ======================================================================
# CHANNEL 1: WHATSAPP CHATBOT 
# ======================================================================


@app.post("/api/whatsapp")
async def whatsapp_webhook(request: Request):
    """
    WhatsApp webhook endpoint for Twilio.

    Intelligently routes between:
    - Voice messages (NumMedia > 0, audio/*)
    - Text messages (Body content)
    """

    try:
        # Get form data
        form_data = await request.form()
        incoming_message = form_data.get("Body", "").strip()
        whatsapp_number = form_data.get("From", "")
        num_media = int(form_data.get("NumMedia", "0"))

        logger.info(f"📱 Received WhatsApp message from {whatsapp_number} (Media: {num_media})")

        # Route to voice handler if media is present
        if num_media > 0:
            media_content_type = form_data.get("MediaContentType0", "")

            # Check if it's audio (voice message)
            if media_content_type.startswith("audio/"):
                logger.info("→ Routing to voice handler")
                return await whatsapp_voice_webhook(request)
            else:
                logger.info(f"→ Non-audio media received: {media_content_type}")

        # Continue with text message handling
        logger.info(f"→ Processing as text: {incoming_message[:50]}...")

        # Skipping signature validation (Twilio validator has bug with integer params)

        if not incoming_message:
            return {"status": "processed"}
        
        db = SessionLocal()
        
        try:
            chatbot = app.state.chatbot
            response_dict = chatbot.generate_response(
                db=db,
                whatsapp_number=whatsapp_number,
                user_message=incoming_message
            )
            
            response_text = response_dict.get("response", "")
            logger.info(f"✓ Generated response: {response_text[:100]}...")
            
            if twilio_client:
                try:
                    message = twilio_client.messages.create(
                        from_=TWILIO_WHATSAPP_NUMBER,
                        body=response_text,
                        to=whatsapp_number
                    )
                    logger.info(f"✓ WhatsApp message sent (SID: {message.sid})")
                    return {
                        "status": "sent",
                        "message_sid": message.sid,
                        "response": response_text
                    }
                except Exception as e:
                    logger.error(f"✗ Failed to send via Twilio: {e}")
                    return {
                        "status": "error",
                        "message": "Failed to send response",
                        "detail": str(e)
                    }
            else:
                logger.warning("⚠️ Twilio not configured")
                return {
                    "status": "processed",
                    "message": response_text,
                    "note": "Twilio not configured"
                }
        
        except Exception as e:
            logger.error(f"✗ Error processing WhatsApp message: {e}", exc_info=True)
            
            if twilio_client:
                try:
                    twilio_client.messages.create(
                        from_=TWILIO_WHATSAPP_NUMBER,
                        body=(
                            "I apologize, but I'm having technical issues.\n"
                            "If you're in crisis, please contact:\n"
                            "- 988 Suicide & Crisis Lifeline\n"
                            "- Crisis Text Line: Text HOME to 741741"
                        ),
                        to=whatsapp_number
                    )
                    logger.info(f"✓ Sent fallback error message to {whatsapp_number}")
                except Exception as send_error:
                    logger.error(f"✗ Failed sending fallback message: {send_error}")
            
            return {"status": "error", "detail": str(e)}
        
        finally:
            db.close()
    
    except Exception as e:
        logger.error(f"✗ WhatsApp webhook error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ======================================================================
# WHATSAPP VOICE MESSAGE WEBHOOK (PHASE 3C)
# ======================================================================


@app.post("/api/whatsapp/voice")
async def whatsapp_voice_webhook(request: Request):
    """
    WhatsApp voice message webhook for Phase 3C.

    Handles voice notes from WhatsApp users:
    1. Downloads audio from Twilio MediaUrl
    2. Transcribes using Azure STT
    3. Processes through LUMI chatbot
    4. Synthesizes response using Azure TTS
    5. Sends audio response back via WhatsApp
    """

    try:
        # Get form data from Twilio
        form_data = await request.form()
        whatsapp_number = form_data.get("From", "")
        num_media = int(form_data.get("NumMedia", "0"))

        logger.info(f"🎤 Received WhatsApp voice message from {whatsapp_number} (Media count: {num_media})")

        # Check if media is present
        if num_media == 0:
            logger.warning("No media attached to WhatsApp message")
            if twilio_client:
                twilio_client.messages.create(
                    from_=TWILIO_WHATSAPP_NUMBER,
                    body="Please send a voice message to talk to LUMI.",
                    to=whatsapp_number
                )
            return {"status": "no_media"}

        # Get media URL and content type
        media_url = form_data.get("MediaUrl0", "")
        media_content_type = form_data.get("MediaContentType0", "")

        logger.info(f"Media URL: {media_url}, Content-Type: {media_content_type}")

        if not media_url:
            logger.error("MediaUrl is missing")
            return {"status": "error", "detail": "No media URL provided"}

        # Check if it's an audio file
        if not media_content_type.startswith("audio/"):
            logger.warning(f"Non-audio media received: {media_content_type}")
            if twilio_client:
                twilio_client.messages.create(
                    from_=TWILIO_WHATSAPP_NUMBER,
                    body="Please send a voice message (not an image or video).",
                    to=whatsapp_number
                )
            return {"status": "invalid_media_type"}

        db = SessionLocal()

        try:
            # Import required services
            import httpx
            from services.speech_to_text import get_stt_service
            from services.text_to_speech import get_tts_service

            # Download audio from Twilio
            logger.info(f"Downloading audio from {media_url}...")
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                # Twilio requires authentication to download media
                # Note: Twilio redirects to CDN (mms.twiliocdn.com), so follow_redirects=True is essential
                auth = (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
                response = await client.get(media_url, auth=auth)
                response.raise_for_status()
                audio_data = response.content

            logger.info(f"✓ Downloaded {len(audio_data)} bytes of audio")

            # Detect audio format from content type
            audio_format = "ogg"  # WhatsApp typically sends OGG
            if "ogg" in media_content_type:
                audio_format = "ogg"
            elif "mp3" in media_content_type:
                audio_format = "mp3"
            elif "wav" in media_content_type:
                audio_format = "wav"
            elif "opus" in media_content_type:
                audio_format = "ogg"  # Opus is in OGG container

            logger.info(f"Detected audio format: {audio_format}")

            # Step 1: Transcribe audio (STT)
            logger.info("Step 1: Transcribing audio...")
            stt_service = get_stt_service()
            transcription_result = stt_service.transcribe(
                audio_data=audio_data,
                audio_format=audio_format,
                language="en-US"
            )

            if not transcription_result.success:
                logger.error(f"Transcription failed: {transcription_result.error_message}")
                if twilio_client:
                    twilio_client.messages.create(
                        from_=TWILIO_WHATSAPP_NUMBER,
                        body="I couldn't understand your voice message. Please try again.",
                        to=whatsapp_number
                    )
                return {"status": "transcription_failed"}

            user_text = transcription_result.text
            logger.info(f"✓ Transcribed: {user_text[:100]}...")

            # Step 2: Process through LUMI chatbot
            logger.info("Step 2: Processing through LUMI chatbot...")
            voice_handler = app.state.voice_handler

            if not voice_handler:
                logger.error("Voice handler not initialized")
                raise Exception("Voice handler not available")

            lumi_result = voice_handler.process_voice_message(
                db=db,
                user_id=whatsapp_number,
                transcribed_text=user_text,
                language_code="en-US",
                user_voice_preference="en-US-AriaNeural"
            )

            if not lumi_result["success"]:
                logger.error(f"LUMI processing failed: {lumi_result.get('error_message')}")
                if twilio_client:
                    twilio_client.messages.create(
                        from_=TWILIO_WHATSAPP_NUMBER,
                        body="I'm having trouble processing your message. Please try again.",
                        to=whatsapp_number
                    )
                return {"status": "processing_failed"}

            bot_response = lumi_result["bot_response"]
            is_crisis = lumi_result.get("is_crisis", False)
            logger.info(f"✓ LUMI response: {bot_response[:100]}...")

            # Step 3: Synthesize response (TTS)
            logger.info("Step 3: Synthesizing audio response...")
            tts_service = get_tts_service()
            synthesis_result = tts_service.synthesize(
                text=bot_response,
                voice="en-US-AriaNeural",
                audio_format="mp3"
            )

            if not synthesis_result.success:
                logger.error(f"Synthesis failed: {synthesis_result.error_message}")
                # Fall back to text response
                if twilio_client:
                    twilio_client.messages.create(
                        from_=TWILIO_WHATSAPP_NUMBER,
                        body=bot_response,
                        to=whatsapp_number
                    )
                return {"status": "synthesis_failed_text_sent"}

            logger.info(f"✓ Synthesized {synthesis_result.duration_seconds:.1f}s of audio")

            # Step 4: Send audio response back via WhatsApp
            logger.info("Step 4: Sending audio response via WhatsApp...")

            if twilio_client:
                try:
                    # Twilio Media API requires a publicly accessible URL
                    # For now, we'll send the text response and note this limitation
                    # In production, you'd upload to Azure Blob Storage or similar

                    # Create message with text (audio sending requires URL)
                    message = twilio_client.messages.create(
                        from_=TWILIO_WHATSAPP_NUMBER,
                        body=bot_response,
                        to=whatsapp_number
                    )

                    logger.info(f"✓ WhatsApp message sent (SID: {message.sid})")
                    logger.warning("⚠️ Audio response generated but not sent - requires media URL hosting")

                    return {
                        "status": "success",
                        "transcription": user_text,
                        "response": bot_response,
                        "is_crisis": is_crisis,
                        "message_sid": message.sid,
                        "note": "Audio generated but sent as text (media hosting not configured)"
                    }

                except Exception as e:
                    logger.error(f"✗ Failed to send via Twilio: {e}")
                    return {
                        "status": "error",
                        "detail": str(e)
                    }
            else:
                logger.warning("⚠️ Twilio not configured")
                return {
                    "status": "processed",
                    "transcription": user_text,
                    "response": bot_response,
                    "note": "Twilio not configured"
                }

        except Exception as e:
            logger.error(f"✗ Error processing WhatsApp voice: {e}", exc_info=True)

            if twilio_client:
                try:
                    twilio_client.messages.create(
                        from_=TWILIO_WHATSAPP_NUMBER,
                        body=(
                            "I'm sorry, I encountered an error processing your voice message. "
                            "Please try again or send a text message instead."
                        ),
                        to=whatsapp_number
                    )
                except Exception as send_error:
                    logger.error(f"✗ Failed sending error message: {send_error}")

            return {"status": "error", "detail": str(e)}

        finally:
            db.close()

    except Exception as e:
        logger.error(f"✗ WhatsApp voice webhook error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ======================================================================
# WHATSAPP STATUS WEBHOOK - HANDLES DELIVERY CONFIRMATIONS
# ======================================================================


@app.post("/api/whatsapp/status")
async def whatsapp_status_webhook(request: Request):
    """Handle WhatsApp message status updates (delivered, read, failed)"""
    try:
        form_data = await request.form()
        message_sid = form_data.get("MessageSid", "")
        message_status = form_data.get("MessageStatus", "")

        logger.info(f"📊 WhatsApp Status - SID: {message_sid}, Status: {message_status}")

        return {"status": "received"}

    except Exception as e:
        logger.error(f"Error in status webhook: {e}")
        return {"status": "error"}


# ======================================================================
# CHANNEL 2: PRIMARY WEB CHAT (WITH CONVERSATION TRACKING)
# ======================================================================


@app.post("/api/webChat")
async def web_chat_tracked(request: Request):
    """Enhanced web chat with conversation tracking."""
    
    try:
        body = await request.json()
        user_message = body.get("content", "").strip()
        conversation_id = body.get("conversation_id")
        message_index = body.get("message_index", 0)
        
        logger.info(
            f"💻 Web chat - Conv: {conversation_id}, "
            f"Msg#{message_index}, Content: {user_message[:50]}..."
        )
        
        if not user_message:
            return JSONResponse({
                "conversation_id": conversation_id,
                "message_index": message_index + 1,
                "content": "Please enter a message.",
                "error": True
            }, status_code=400)
        
        db = SessionLocal()
        
        try:
            chatbot = app.state.chatbot
            
            # If new conversation
            if not conversation_id:
                conversation_id = str(uuid.uuid4())
                logger.info(f"✓ New conversation created: {conversation_id}")
            
            web_user_id = f"web:{conversation_id}"
            
            response_dict = chatbot.generate_response(
                db=db,
                whatsapp_number=web_user_id,
                user_message=user_message
            )
            
            response_text = response_dict.get("response", "")
            is_crisis = response_dict.get("is_crisis", False)
            
            logger.info(
                f"✓ Web chat response - Conv: {conversation_id}, "
                f"Msg#{message_index + 1}"
            )
            
            return JSONResponse({
                "conversation_id": conversation_id,
                "message_index": message_index + 1,
                "content": response_text,
                "is_crisis": is_crisis
            })
        
        except Exception as e:
            logger.error(f"✗ Web chat processing error: {e}", exc_info=True)
            return JSONResponse({
                "conversation_id": conversation_id,
                "message_index": message_index + 1,
                "content": "I apologize, something went wrong.",
                "error": True
            }, status_code=500)
        
        finally:
            db.close()
    
    except Exception as e:
        logger.error(f"✗ WebChat endpoint error: {e}", exc_info=True)
        return JSONResponse({
            "conversation_id": None,
            "message_index": 0,
            "content": "Sorry, something went wrong.",
            "error": True
        }, status_code=500)


# ======================================================================
# CHANNEL 2B: SIMPLE WEB CHAT (BACKWARD COMPATIBLE)
# ======================================================================


@app.post("/api/awe-chat")
async def awe_chat(request: Request):
    """Simple web chat endpoint without conversation tracking."""
    
    try:
        body = await request.json()
        user_message = body.get("message", "").strip()
        user_id = f"web-{datetime.utcnow().timestamp()}"
        
        logger.info(f"💻 Simple chat - Message: {user_message[:50]}...")
        
        if not user_message:
            return JSONResponse(
                {"reply": "Please enter a message.", "error": True},
                status_code=400
            )
        
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
            
            return JSONResponse({
                "reply": response_text,
                "is_crisis": is_crisis,
                "user_id": user_id
            })
        
        except Exception as e:
            logger.error(f"✗ Simple chat processing error: {e}", exc_info=True)
            return JSONResponse({
                "reply": "I apologize, something went wrong.",
                "error": True
            }, status_code=500)
        
        finally:
            db.close()
    
    except Exception as e:
        logger.error(f"✗ Simple chat endpoint error: {e}", exc_info=True)
        return JSONResponse({
            "reply": "Sorry, something went wrong.",
            "error": True
        }, status_code=500)


# ======================================================================
# TESTING ENDPOINT
# ======================================================================


@app.post("/api/test-message")
async def test_message(message: dict):
    """Quick testing endpoint"""
    
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


# ======================================================================
# LIVE DASHBOARD WITH API KEY PROTECTION
# ======================================================================
# Real-time dashboard + 5 API endpoints + Key-based security


DASHBOARD_SECRET_KEY = "AWE-LUMI@2025!Logs"


# Dashboard serving endpoint (HTML page with key protection)
@app.get("/dashboard")
async def serve_dashboard(key: str = None):
    """
    Serve dashboard HTML with API key protection
    Access: [http://yourdomain.com/dashboard?key=YOUR-SECRET-KEY](http://yourdomain.com/dashboard?key=YOUR-SECRET-KEY)
    """
    if key != DASHBOARD_SECRET_KEY:
        return {
            "error": "Invalid or missing API key",
            "message": "Access denied"
        }, 403
    
    try:
        with open("dashboard.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return {"error": "Dashboard file not found"}, 404


# ======================================================================
# DASHBOARD METRICS API ENDPOINTS (5 Total)
# ======================================================================


# API endpoint 1: Get all dashboard metrics
@app.get("/api/dashboard/metrics")
async def get_dashboard_metrics():
    """
    Get all dashboard metrics from database
    Returns: WhatsApp users, Web users, Total users, Messages, Crisis alerts, Active today
    """
    db = SessionLocal()
    try:
        from database import aichatusers, Message
        
        # Count by source_channel column
        try:
            whatsapp_count = db.query(aichatusers).filter(
                aichatusers.source_channel == 'whatsapp'
            ).count()
            
            web_count = db.query(aichatusers).filter(
                aichatusers.source_channel == 'web'
            ).count()
        except:
            # Fallback if source_channel doesn't exist yet
            logger.warning("⚠️ source_channel column not found, using fallback method")
            whatsapp_count = db.query(aichatusers).filter(
                aichatusers.whatsapp_number.isnot(None)
            ).count()
            total_count = db.query(aichatusers).count()
            web_count = total_count - whatsapp_count
        
        total_count = db.query(aichatusers).count()

        # Fix: Sum all user message counts instead of counting Message table rows
        total_messages = db.query(func.sum(aichatusers.total_messages)).scalar() or 0

        crisis_count = db.query(aichatusers).filter(
            aichatusers.crisis_flag == True
        ).count()
        
        active_today = db.query(aichatusers).filter(
            aichatusers.last_interaction > datetime.utcnow() - timedelta(hours=24)
        ).count()
        
        return {
            'whatsapp_users': whatsapp_count,
            'web_users': web_count,
            'total_users': total_count,
            'total_messages': total_messages,
            'crisis_alerts': crisis_count,
            'active_today': active_today,
            'last_updated': datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error fetching dashboard metrics: {e}")
        return {'error': str(e)}, 500
    finally:
        db.close()


# API endpoint 2: Get 30-day user growth
@app.get("/api/dashboard/growth")
async def get_user_growth():
    """
    Get 30-day user growth data
    Returns: Daily cumulative user count for trend analysis
    """
    db = SessionLocal()
    try:
        from database import aichatusers
        
        growth_data = []
        
        for i in range(30, -1, -1):
            date = datetime.utcnow() - timedelta(days=i)
            date_start = date.replace(hour=0, minute=0, second=0, microsecond=0)
            count = db.query(aichatusers).filter(
                aichatusers.created_at <= date_start
            ).count()
            growth_data.append({
                'date': date_start.strftime('%b %d'),
                'users': count
            })
        
        return {'data': growth_data}
    except Exception as e:
        logger.error(f"Error fetching growth data: {e}")
        return {'error': str(e)}, 500
    finally:
        db.close()


# API endpoint 3: Get crisis alerts by day of week
@app.get("/api/dashboard/crisis-by-day")
async def get_crisis_by_day():
    """
    Get crisis alerts grouped by day of week
    Returns: Crisis count for each day (Monday-Sunday)
    """
    db = SessionLocal()
    try:
        from database import Message
        
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        crisis_by_day = {}
        
        for day_num, day_name in enumerate(days):
            count = db.query(Message).filter(
                Message.contains_crisis_keywords == True,
                extract('dow', Message.timestamp) == day_num
            ).count()
            crisis_by_day[day_name] = count
        
        return {'crisis_by_day': crisis_by_day}
    except Exception as e:
        logger.error(f"Error fetching crisis data: {e}")
        return {'error': str(e)}, 500
    finally:
        db.close()


# API endpoint 4: Get WhatsApp vs Web distribution
@app.get("/api/dashboard/source-distribution")
async def get_source_distribution():
    """
    Get user distribution by source (WhatsApp vs Web)
    Returns: Percentage and count breakdown
    """
    db = SessionLocal()
    try:
        from database import aichatusers
        
        # Try with source_channel first
        try:
            whatsapp_count = db.query(aichatusers).filter(
                aichatusers.source_channel == 'whatsapp'
            ).count()
            
            web_count = db.query(aichatusers).filter(
                aichatusers.source_channel == 'web'
            ).count()
        except:
            # Fallback if source_channel doesn't exist yet
            logger.warning("⚠️ source_channel column not found, using fallback method")
            whatsapp_count = db.query(aichatusers).filter(
                aichatusers.whatsapp_number.isnot(None)
            ).count()
            total_count = db.query(aichatusers).count()
            web_count = total_count - whatsapp_count
        
        total_count = db.query(aichatusers).count()
        
        if total_count == 0:
            return {
                'whatsapp_percent': 0,
                'web_percent': 0,
                'whatsapp_count': 0,
                'web_count': 0
            }
        
        return {
            'whatsapp_percent': round((whatsapp_count / total_count) * 100, 1),
            'web_percent': round((web_count / total_count) * 100, 1),
            'whatsapp_count': whatsapp_count,
            'web_count': web_count
        }
    except Exception as e:
        logger.error(f"Error fetching source distribution: {e}")
        return {'error': str(e)}, 500
    finally:
        db.close()


# API endpoint 5: Get active users (last 7 days)
@app.get("/api/dashboard/active-users")
async def get_active_users():
    """
    Get active users for each day (last 7 days)
    Returns: Number of unique active users per day
    """
    db = SessionLocal()
    try:
        from database import aichatusers
        
        active_data = []
        days_short = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
        
        for i in range(6, -1, -1):
            date = datetime.utcnow() - timedelta(days=i)
            date_start = date.replace(hour=0, minute=0, second=0, microsecond=0)
            date_end = date_start + timedelta(hours=24)
            
            count = db.query(aichatusers).filter(
                aichatusers.last_interaction >= date_start,
                aichatusers.last_interaction < date_end
            ).count()
            
            day_name = days_short[date_start.weekday()]
            active_data.append({
                'day': day_name,
                'users': count
            })
        
        return {'data': active_data}
    except Exception as e:
        logger.error(f"Error fetching active users: {e}")
        return {'error': str(e)}, 500
    finally:
        db.close()


# ======================================================================
# VOICE INTEGRATION ROUTES
# ======================================================================
VOICE_ENABLED = os.getenv("VOICE_ENABLED", "true").lower() == "true"

if VOICE_ROUTES_AVAILABLE and VOICE_ENABLED:
    app.include_router(voice_router)
    logger.info("✅ Voice routes enabled and loaded")
elif VOICE_ROUTES_AVAILABLE and not VOICE_ENABLED:
    logger.info("⚠️ Voice routes available but VOICE_ENABLED=false")


# ======================================================================
# ERROR HANDLER
# ======================================================================


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Internal server error",
            "detail": str(exc)
        }
    )


# ======================================================================
# RUN SERVER
# ======================================================================


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8001)),
        log_level="info"
    )

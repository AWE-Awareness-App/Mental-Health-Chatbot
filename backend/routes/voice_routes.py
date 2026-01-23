"""
Voice API Routes for AWE Mental Health Chatbot

Provides REST API endpoints for voice integration:
- POST /api/voice/transcribe - Speech-to-Text
- POST /api/voice/synthesize - Text-to-Speech  
- POST /api/voice/process - Complete voice conversation flow
- GET /api/voice/status - Voice service health check
- GET /api/voice/voices - List available voices

Author: Emergent
Version: 1.0.0
"""

import os
import time
import logging
import base64
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, File, UploadFile, Form, Request, Depends
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from services.speech_to_text import get_stt_service, SpeechToTextService
from services.text_to_speech import get_tts_service, TextToSpeechService
from services.voice_chatbot_handler import VoiceChatbotHandler
from utils.audio_utils import (
    validate_audio,
    AudioFileHandler,
    get_mime_type,
    audio_to_base64,
    base64_to_audio,
)

logger = logging.getLogger(__name__)

# Create router with /api/voice prefix
router = APIRouter(prefix="/api/voice", tags=["Voice"])

# Configuration from environment
VOICE_ENABLED = os.getenv("VOICE_ENABLED", "true").lower() == "true"
VOICE_RATE_LIMIT_ENABLED = os.getenv("VOICE_RATE_LIMIT_ENABLED", "true").lower() == "true"
MAX_AUDIO_SIZE_MB = float(os.getenv("MAX_AUDIO_SIZE", "10"))
MAX_AUDIO_DURATION = float(os.getenv("MAX_AUDIO_DURATION", "60"))

# Initialize audio handler
audio_handler = AudioFileHandler(
    max_size_mb=MAX_AUDIO_SIZE_MB,
    max_duration_seconds=MAX_AUDIO_DURATION
)


# ============================================================================
# Request/Response Models
# ============================================================================

class SynthesizeRequest(BaseModel):
    """Request body for text-to-speech synthesis."""
    text: str = Field(..., min_length=1, max_length=5000, description="Text to synthesize")
    voice: Optional[str] = Field(None, description="Voice name (e.g., 'en-US-AriaNeural')")
    format: Optional[str] = Field("mp3", description="Audio output format (mp3, wav)")
    return_base64: Optional[bool] = Field(True, description="Return audio as base64 string")


class TranscriptionResponse(BaseModel):
    """Response from transcription endpoint."""
    success: bool
    text: str
    confidence: float
    language: str
    duration_seconds: float
    error_message: Optional[str] = None


class SynthesisResponse(BaseModel):
    """Response from synthesis endpoint."""
    success: bool
    audio_format: str
    duration_seconds: float
    voice_used: str
    audio_base64: Optional[str] = None
    error_message: Optional[str] = None


class VoiceProcessRequest(BaseModel):
    """Request for complete voice processing flow with base64 audio."""
    audio_base64: str = Field(..., description="Base64 encoded audio data")
    audio_format: Optional[str] = Field("wav", description="Audio format")
    user_id: Optional[str] = Field(None, description="User identifier for rate limiting")
    language: Optional[str] = Field(None, description="Language code (e.g., 'en-US')")
    voice: Optional[str] = Field(None, description="Voice for TTS response")
    include_audio_response: Optional[bool] = Field(True, description="Include TTS audio in response")


class VoiceProcessResponse(BaseModel):
    """Response from complete voice processing."""
    success: bool
    user_transcription: str
    bot_response: str
    bot_audio_base64: Optional[str] = None
    audio_duration_seconds: float
    sources: Optional[list] = None
    is_crisis: Optional[bool] = False
    processing_time_seconds: float
    error_message: Optional[str] = None


class VoiceStatusResponse(BaseModel):
    """Response from voice status endpoint."""
    voice_enabled: bool
    stt_available: bool
    stt_status: str
    tts_available: bool
    tts_status: str
    service_type: str
    supported_languages: list
    configuration: Dict[str, Any]


class VoiceInfo(BaseModel):
    """Voice information."""
    name: str
    display_name: str
    language: str
    gender: str
    neural: bool


# ============================================================================
# Dependency: Check if voice is enabled
# ============================================================================

def check_voice_enabled():
    """Dependency to check if voice features are enabled."""
    if not VOICE_ENABLED:
        raise HTTPException(
            status_code=503,
            detail="Voice features are currently disabled. Set VOICE_ENABLED=true to enable."
        )
    return True


# ============================================================================
# Rate Limiter Integration (Optional)
# ============================================================================

class VoiceRateLimiter:
    """
    Simple rate limiter for voice endpoints.
    
    In production, this integrates with the existing rate_limiter.py service.
    """
    
    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: Dict[str, list] = {}
    
    def is_allowed(self, user_id: str) -> bool:
        """Check if user is allowed to make request."""
        if not VOICE_RATE_LIMIT_ENABLED:
            return True
        
        current_time = time.time()
        user_requests = self._requests.get(user_id, [])
        
        # Remove old requests
        user_requests = [t for t in user_requests if current_time - t < self.window_seconds]
        self._requests[user_id] = user_requests
        
        if len(user_requests) >= self.max_requests:
            return False
        
        user_requests.append(current_time)
        return True
    
    def get_retry_after(self, user_id: str) -> int:
        """Get seconds until rate limit resets."""
        current_time = time.time()
        user_requests = self._requests.get(user_id, [])
        
        if not user_requests:
            return 0
        
        oldest = min(user_requests)
        retry_after = int(self.window_seconds - (current_time - oldest))
        return max(0, retry_after)


# Initialize rate limiter
voice_rate_limiter = VoiceRateLimiter(max_requests=10, window_seconds=60)


def check_rate_limit(user_id: Optional[str] = None):
    """
    Check rate limit for user.
    
    Args:
        user_id: User identifier (from request or form)
    
    Raises:
        HTTPException 429 if rate limited
    """
    if not VOICE_RATE_LIMIT_ENABLED:
        return
    
    if not user_id:
        user_id = "anonymous"
    
    if not voice_rate_limiter.is_allowed(user_id):
        retry_after = voice_rate_limiter.get_retry_after(user_id)
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Try again in {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)}
        )


# ============================================================================
# ENDPOINT: GET /api/voice/status
# ============================================================================

@router.get("/status", response_model=VoiceStatusResponse)
async def voice_status():
    """
    Check voice service health and configuration.
    
    Returns:
        VoiceStatusResponse with service availability and configuration
    """
    stt_service = get_stt_service()
    tts_service = get_tts_service()
    
    stt_available, stt_status = stt_service.is_available()
    tts_available, tts_status = tts_service.is_available()
    
    return VoiceStatusResponse(
        voice_enabled=VOICE_ENABLED,
        stt_available=stt_available,
        stt_status=stt_status,
        tts_available=tts_available,
        tts_status=tts_status,
        service_type=os.getenv("VOICE_SERVICE", "mock"),
        supported_languages=stt_service.get_supported_languages(),
        configuration={
            "max_audio_size_mb": MAX_AUDIO_SIZE_MB,
            "max_audio_duration_seconds": MAX_AUDIO_DURATION,
            "rate_limit_enabled": VOICE_RATE_LIMIT_ENABLED,
            "rate_limit_max_requests": 10,
            "rate_limit_window_seconds": 60
        }
    )


# ============================================================================
# ENDPOINT: GET /api/voice/voices
# ============================================================================

@router.get("/voices", response_model=list)
async def list_voices(language: Optional[str] = None):
    """
    List available voices for text-to-speech.
    
    Args:
        language: Optional language filter (e.g., 'en' for all English voices)
    
    Returns:
        List of available voice options
    """
    tts_service = get_tts_service()
    voices = tts_service.get_available_voices(language_filter=language)
    return [v.to_dict() for v in voices]


# ============================================================================
# ENDPOINT: POST /api/voice/transcribe
# ============================================================================

@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    audio: UploadFile = File(..., description="Audio file to transcribe"),
    user_id: Optional[str] = Form(None, description="User ID for rate limiting"),
    language: Optional[str] = Form(None, description="Language code (e.g., 'en-US')"),
    _enabled: bool = Depends(check_voice_enabled)
):
    """
    Transcribe audio file to text using Speech-to-Text.
    
    Accepts audio file upload (WAV, MP3, OGG, WebM, M4A).
    Maximum file size: 10MB
    Maximum duration: 60 seconds
    
    Args:
        audio: Audio file upload
        user_id: Optional user identifier for rate limiting
        language: Optional language code for recognition
    
    Returns:
        TranscriptionResponse with transcribed text and metadata
    
    Raises:
        400: Invalid audio file
        413: Audio file too large
        422: Audio duration too long
        429: Rate limit exceeded
        503: Voice service unavailable
    """
    start_time = time.time()
    
    # Check rate limit
    check_rate_limit(user_id)
    
    logger.info(f"Transcribe request - User: {user_id}, Filename: {audio.filename}")
    
    try:
        # Read and validate audio
        audio_data, validation = await audio_handler.process_upload(audio)
        
        if not validation.is_valid:
            # Determine appropriate error code
            error_msg = validation.error_message or "Invalid audio file"
            
            if "size" in error_msg.lower() and "exceeds" in error_msg.lower():
                raise HTTPException(status_code=413, detail=error_msg)
            elif "duration" in error_msg.lower() and "exceeds" in error_msg.lower():
                raise HTTPException(status_code=422, detail=error_msg)
            else:
                raise HTTPException(status_code=400, detail=error_msg)
        
        # Get format from validation or filename
        audio_format = validation.format
        if audio_format == "unknown" and audio.filename:
            audio_format = audio_handler.get_format_from_filename(audio.filename)
        
        # Transcribe
        stt_service = get_stt_service()
        result = stt_service.transcribe(audio_data, audio_format, language)
        
        if not result.success:
            logger.warning(f"Transcription failed: {result.error_message}")
            raise HTTPException(
                status_code=503 if "not initialized" in (result.error_message or "").lower() else 500,
                detail=result.error_message or "Transcription failed"
            )
        
        elapsed = time.time() - start_time
        logger.info(f"Transcription successful in {elapsed:.2f}s - Text: {result.text[:50]}...")
        
        return TranscriptionResponse(
            success=True,
            text=result.text,
            confidence=result.confidence,
            language=result.language,
            duration_seconds=result.duration_seconds
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


# ============================================================================
# ENDPOINT: POST /api/voice/synthesize
# ============================================================================

@router.post("/synthesize")
async def synthesize_speech(
    request: SynthesizeRequest,
    _enabled: bool = Depends(check_voice_enabled)
):
    """
    Synthesize text to speech audio.
    
    Args:
        request: SynthesizeRequest with text and options
    
    Returns:
        Audio file (if return_base64=False) or JSON with base64 audio
    
    Raises:
        400: Invalid request
        503: Voice service unavailable
    """
    start_time = time.time()
    
    logger.info(f"Synthesize request - Voice: {request.voice}, Format: {request.format}, Text length: {len(request.text)}")
    
    try:
        tts_service = get_tts_service()
        result = tts_service.synthesize(
            text=request.text,
            voice=request.voice,
            audio_format=request.format or "mp3"
        )
        
        if not result.success:
            logger.warning(f"Synthesis failed: {result.error_message}")
            raise HTTPException(
                status_code=503 if "not initialized" in (result.error_message or "").lower() else 500,
                detail=result.error_message or "Speech synthesis failed"
            )
        
        elapsed = time.time() - start_time
        logger.info(f"Synthesis successful in {elapsed:.2f}s - Duration: {result.duration_seconds:.1f}s")
        
        # Return as base64 JSON or raw audio
        if request.return_base64:
            return JSONResponse({
                "success": True,
                "audio_base64": result.audio_base64,
                "audio_format": result.audio_format,
                "duration_seconds": result.duration_seconds,
                "voice_used": result.voice_used
            })
        else:
            # Return raw audio file
            return Response(
                content=result.audio_data,
                media_type=get_mime_type(result.audio_format),
                headers={
                    "Content-Disposition": f"attachment; filename=speech.{result.audio_format}",
                    "X-Audio-Duration": str(result.duration_seconds),
                    "X-Voice-Used": result.voice_used
                }
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Synthesis error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


# ============================================================================
# Voice Handler Global Instance
# ============================================================================

# Global voice handler (initialized on first use or from server.py)
_voice_handler: Optional[VoiceChatbotHandler] = None


def get_voice_handler() -> Optional[VoiceChatbotHandler]:
    """Get or create voice handler instance."""
    global _voice_handler
    return _voice_handler


def set_voice_handler(handler: VoiceChatbotHandler):
    """Set voice handler instance (called from server.py)."""
    global _voice_handler
    _voice_handler = handler
    logger.info("Voice handler set successfully")


# ============================================================================
# ENDPOINT: POST /api/voice/process
# ============================================================================

@router.post("/process", response_model=VoiceProcessResponse)
async def process_voice_conversation(
    request: Request,
    audio: Optional[UploadFile] = File(None, description="Audio file to process"),
    audio_base64: Optional[str] = Form(None, description="Base64 encoded audio (alternative to file)"),
    user_id: Optional[str] = Form(None, description="User ID for rate limiting and tracking"),
    language: Optional[str] = Form(None, description="Language code (e.g., 'en-US')"),
    voice: Optional[str] = Form(None, description="Voice for TTS response"),
    include_audio_response: bool = Form(True, description="Include TTS audio in response"),
    _enabled: bool = Depends(check_voice_enabled)
):
    """
    Complete voice conversation flow:
    1. Transcribe user's audio to text (STT)
    2. Process through LUMI chatbot (RAG + crisis detection + therapy)
    3. Synthesize response to audio (TTS)

    This is the main endpoint for end-to-end voice conversations.

    Args:
        audio: Audio file upload (WAV, MP3, OGG)
        audio_base64: Base64 encoded audio (alternative to file upload)
        user_id: User identifier for rate limiting
        language: Language code for STT
        voice: Voice name for TTS response
        include_audio_response: Whether to include TTS audio in response

    Returns:
        VoiceProcessResponse with transcription, bot response, and optional audio

    Raises:
        400: No audio provided or invalid audio
        413: Audio too large
        422: Audio too long
        429: Rate limit exceeded
        503: Voice service unavailable
    """
    import uuid as uuid_module

    start_time = time.time()

    # Check rate limit
    check_rate_limit(user_id)

    logger.info(f"Voice process request - User: {user_id}")

    try:
        # Get audio data from file or base64
        if audio is not None:
            audio_data, validation = await audio_handler.process_upload(audio)
            audio_format = validation.format
        elif audio_base64:
            try:
                audio_data = base64_to_audio(audio_base64)
                validation = validate_audio(
                    audio_data,
                    max_size_mb=MAX_AUDIO_SIZE_MB,
                    max_duration_seconds=MAX_AUDIO_DURATION
                )
                audio_format = validation.format
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
        else:
            raise HTTPException(
                status_code=400,
                detail="No audio provided. Send either 'audio' file or 'audio_base64' string."
            )

        # Validate audio
        if not validation.is_valid:
            error_msg = validation.error_message or "Invalid audio file"
            if "size" in error_msg.lower() and "exceeds" in error_msg.lower():
                raise HTTPException(status_code=413, detail=error_msg)
            elif "duration" in error_msg.lower() and "exceeds" in error_msg.lower():
                raise HTTPException(status_code=422, detail=error_msg)
            else:
                raise HTTPException(status_code=400, detail=error_msg)

        # Step 1: Transcribe audio
        stt_service = get_stt_service()
        transcription = stt_service.transcribe(audio_data, audio_format, language)

        if not transcription.success:
            raise HTTPException(
                status_code=503,
                detail=f"Transcription failed: {transcription.error_message}"
            )

        user_text = transcription.text
        logger.info(f"Transcribed: {user_text[:100]}...")

        # Step 2: Process through LUMI chatbot
        voice_handler = get_voice_handler()
        bot_response_text = ""
        sources = []
        is_crisis = False

        if voice_handler is not None:
            # Get database session from app state
            from database import SessionLocal

            db = SessionLocal()
            try:
                # Get user's voice preference from database
                from database import UserVoicePreference

                user_voice_pref = None
                if user_id:
                    user_voice_pref = db.query(UserVoicePreference).filter(
                        UserVoicePreference.user_id == user_id
                    ).first()

                preferred_voice = user_voice_pref.preferred_voice if user_voice_pref else voice

                # Process through LUMI
                response_dict = voice_handler.process_voice_message(
                    db=db,
                    user_id=user_id or f"voice-{uuid_module.uuid4()}",
                    transcribed_text=user_text,
                    language_code=language or "en-US",
                    user_voice_preference=preferred_voice,
                )

                if not response_dict["success"]:
                    # Fall back to placeholder if LUMI fails
                    logger.warning(f"LUMI processing failed: {response_dict['error_message']}")
                    bot_response_text = _get_placeholder_response(user_text)
                    sources = []
                else:
                    bot_response_text = response_dict["bot_response"]
                    sources = response_dict["sources"]
                    is_crisis = response_dict["is_crisis"]
                    # Use voice from LUMI if available
                    if response_dict.get("suggested_voice"):
                        voice = response_dict["suggested_voice"]

            finally:
                db.close()
        else:
            # Fallback: Voice handler not initialized, use placeholder
            logger.warning("Voice handler not initialized, using placeholder response")
            bot_response_text = _get_placeholder_response(user_text)
            sources = ["Voice handler not initialized - using placeholder"]

        logger.info(f"Bot response: {bot_response_text[:100]}...")

        # Step 3: Synthesize response (optional)
        bot_audio_base64 = None
        audio_duration = 0.0

        if include_audio_response:
            tts_service = get_tts_service()
            synthesis = tts_service.synthesize(
                text=bot_response_text,
                voice=voice,
                audio_format="mp3"
            )

            if synthesis.success:
                bot_audio_base64 = synthesis.audio_base64
                audio_duration = synthesis.duration_seconds
            else:
                logger.warning(f"TTS failed (non-critical): {synthesis.error_message}")

        elapsed = time.time() - start_time
        logger.info(f"Voice process completed in {elapsed:.2f}s")

        return VoiceProcessResponse(
            success=True,
            user_transcription=user_text,
            bot_response=bot_response_text,
            bot_audio_base64=bot_audio_base64,
            audio_duration_seconds=audio_duration,
            sources=sources,
            is_crisis=is_crisis,
            processing_time_seconds=elapsed
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice process error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


def _get_placeholder_response(user_text: str) -> str:
    """
    Generate placeholder response for testing.
    
    In production, replace this with actual chatbot integration.
    """
    # Simple keyword-based response for testing
    text_lower = user_text.lower()
    
    if any(word in text_lower for word in ["hello", "hi", "hey"]):
        return "Hello! I'm here to support your wellness journey. How can I help you today?"
    
    if any(word in text_lower for word in ["screen time", "phone", "social media"]):
        return (
            "I understand managing screen time can be challenging. Let's start with awareness. "
            "What usually triggers you to reach for your phone? Understanding our patterns is "
            "the first step in the Four Aces framework."
        )
    
    if any(word in text_lower for word in ["anxious", "anxiety", "worried", "stress"]):
        return (
            "I hear that you're feeling stressed. Take a moment to breathe deeply. "
            "Remember, you have more control than you might think right now. "
            "What's one small thing within your control that you could focus on today?"
        )
    
    if any(word in text_lower for word in ["happy", "happiness", "joy"]):
        return (
            "Happiness is a wonderful goal! The Beyond Happy framework teaches us that "
            "happiness is a way of being, not a destination. What brings you moments "
            "of joy or contentment in your daily life?"
        )
    
    # Default response
    return (
        "Thank you for sharing. I'm here to support your digital wellness journey. "
        "What aspect of your well-being would you like to explore today?"
    )


# ============================================================================
# ENDPOINT: POST /api/voice/process-json (Alternative JSON-based endpoint)
# ============================================================================

@router.post("/process-json", response_model=VoiceProcessResponse)
async def process_voice_json(
    request: VoiceProcessRequest,
    _enabled: bool = Depends(check_voice_enabled)
):
    """
    Complete voice conversation flow using JSON body with base64 audio.
    
    Alternative to the multipart form endpoint for clients that prefer JSON.
    
    Args:
        request: VoiceProcessRequest with base64 audio and options
    
    Returns:
        VoiceProcessResponse with transcription, bot response, and optional audio
    """
    start_time = time.time()
    
    # Check rate limit
    check_rate_limit(request.user_id)
    
    logger.info(f"Voice process (JSON) request - User: {request.user_id}")
    
    try:
        # Decode audio
        try:
            audio_data = base64_to_audio(request.audio_base64)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 audio: {str(e)}")
        
        # Validate
        validation = validate_audio(
            audio_data,
            max_size_mb=MAX_AUDIO_SIZE_MB,
            max_duration_seconds=MAX_AUDIO_DURATION
        )
        
        if not validation.is_valid:
            error_msg = validation.error_message or "Invalid audio"
            if "size" in error_msg.lower() and "exceeds" in error_msg.lower():
                raise HTTPException(status_code=413, detail=error_msg)
            elif "duration" in error_msg.lower() and "exceeds" in error_msg.lower():
                raise HTTPException(status_code=422, detail=error_msg)
            else:
                raise HTTPException(status_code=400, detail=error_msg)
        
        # Transcribe
        stt_service = get_stt_service()
        transcription = stt_service.transcribe(
            audio_data,
            request.audio_format or validation.format,
            request.language
        )
        
        if not transcription.success:
            raise HTTPException(
                status_code=503,
                detail=f"Transcription failed: {transcription.error_message}"
            )
        
        user_text = transcription.text
        
        # Generate response (placeholder)
        bot_response_text = _get_placeholder_response(user_text)
        sources = ["Placeholder - integrate with your chatbot pipeline"]
        
        # Synthesize response
        bot_audio_base64 = None
        audio_duration = 0.0
        
        if request.include_audio_response:
            tts_service = get_tts_service()
            synthesis = tts_service.synthesize(
                text=bot_response_text,
                voice=request.voice,
                audio_format="mp3"
            )
            
            if synthesis.success:
                bot_audio_base64 = synthesis.audio_base64
                audio_duration = synthesis.duration_seconds
        
        elapsed = time.time() - start_time
        
        return VoiceProcessResponse(
            success=True,
            user_transcription=user_text,
            bot_response=bot_response_text,
            bot_audio_base64=bot_audio_base64,
            audio_duration_seconds=audio_duration,
            sources=sources,
            processing_time_seconds=elapsed
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice process (JSON) error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


# ============================================================================
# ENDPOINT: POST /api/voice/preferences
# ============================================================================

@router.post("/preferences")
async def save_voice_preferences(user_id: str, request: Request):
    """Save user's voice and language preferences."""

    try:
        body = await request.json()

        from database import SessionLocal, UserVoicePreference

        db = SessionLocal()
        try:
            # Get or create preference record
            pref = db.query(UserVoicePreference).filter(
                UserVoicePreference.user_id == user_id
            ).first()

            if not pref:
                pref = UserVoicePreference(user_id=user_id)
                db.add(pref)

            # Update preferences
            if "preferred_voice" in body:
                pref.preferred_voice = body["preferred_voice"]

            if "preferred_language" in body:
                pref.preferred_language = body["preferred_language"]

            if "preferred_voice_gender" in body:
                pref.preferred_voice_gender = body["preferred_voice_gender"]

            if "auto_play_audio" in body:
                pref.auto_play_audio = body["auto_play_audio"]

            if "audio_speed" in body:
                pref.audio_speed = body["audio_speed"]

            db.commit()

            return {"success": True, "message": "Preferences saved"}

        finally:
            db.close()

    except Exception as e:
        logger.error(f"Error saving preferences: {e}")
        return {"success": False, "error_message": str(e)}


# ============================================================================
# ENDPOINT: GET /api/voice/preferences/{user_id}
# ============================================================================

@router.get("/preferences/{user_id}")
async def get_voice_preferences(user_id: str):
    """Get user's voice preferences."""

    try:
        from database import SessionLocal, UserVoicePreference

        db = SessionLocal()
        try:
            pref = db.query(UserVoicePreference).filter(
                UserVoicePreference.user_id == user_id
            ).first()

            if not pref:
                # Return defaults
                return {
                    "user_id": user_id,
                    "preferred_voice": "en-US-AriaNeural",
                    "preferred_voice_gender": "Female",
                    "preferred_language": "en-US",
                    "auto_play_audio": True,
                    "audio_speed": 1.0,
                }

            return {
                "user_id": user_id,
                "preferred_voice": pref.preferred_voice,
                "preferred_voice_gender": pref.preferred_voice_gender,
                "preferred_language": pref.preferred_language,
                "auto_play_audio": pref.auto_play_audio,
                "audio_speed": pref.audio_speed,
            }

        finally:
            db.close()

    except Exception as e:
        logger.error(f"Error getting preferences: {e}")
        return {"success": False, "error_message": str(e)}

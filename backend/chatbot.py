"""
Therapeutic chatbot logic with OpenAI GPT-4 integration - MVP VERSION.
Handles response generation, crisis detection, and conversation management.

ENHANCEMENTS:
- Robust error handling with graceful degradation
- Fallback responses when APIs fail
- Better logging and metrics tracking
"""

import os
import re
from typing import List, Dict, Optional
from openai import OpenAI, OpenAIError, RateLimitError, APIError
from sqlalchemy.orm import Session
from rag_system_v2 import TherapeuticRAG
from database import (
    get_or_create_user, 
    get_active_conversation, 
    save_message,
    get_conversation_history
)
import logging
import time

logger = logging.getLogger(__name__)


class TherapeuticChatbot:
    """Main chatbot class for handling therapeutic conversations."""
    
    # Crisis keywords that trigger immediate intervention
    CRISIS_KEYWORDS = [
        "suicide", "suicidal", "kill myself", "end my life", "want to die",
        "self-harm", "hurt myself", "cutting", "overdose", "no reason to live",
        "better off dead", "hopeless", "can't go on"
    ]
    
    # Crisis response template
    CRISIS_RESPONSE = """I hear that you're going through an extremely difficult time, and I'm concerned about your safety. Your life matters, and there are people who want to help you right now.

**Please reach out for immediate support:**
🆘 National Suicide Prevention Lifeline: 988 (call or text)
💬 Crisis Text Line: Text HOME to 741741
🌐 SAMHSA National Helpline: 1-800-662-4357

These services are free, confidential, and available 24/7. Trained counselors are ready to listen and help.

If you're in immediate danger, please call 911 or go to your nearest emergency room.

I'm here to support you with digital wellness, but professional crisis counselors are better equipped to help with these intense feelings. Would you like to talk about what's bringing you to reach out today?"""
    
    def __init__(self, openai_api_key: str):
        """Initialize chatbot with OpenAI client and RAG system."""
        self.client = OpenAI(api_key=openai_api_key)
        self.rag = TherapeuticRAG(openai_api_key)
        self.model = "gpt-4-turbo-preview"  # Using GPT-4 Turbo for better responses
    
    def detect_crisis(self, message: str) -> bool:
        """Detect if message contains crisis-related keywords."""
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in self.CRISIS_KEYWORDS)
    
    def generate_response(self, db: Session, whatsapp_number: str, 
                         user_message: str) -> Dict:
        """
        Generate therapeutic response using RAG and GPT-4 with robust error handling.
        
        Returns:
            Dict with 'response', 'is_crisis', and 'user_id'
        """
        start_time = time.time()
        
        try:
            # Get or create user
            user = get_or_create_user(db, whatsapp_number)
            
            # Check for crisis content
            is_crisis = self.detect_crisis(user_message)
            
            if is_crisis:
                logger.warning(f"⚠️ Crisis content detected from {whatsapp_number}")
                user.crisis_flag = True
                db.commit()
                
                # Get or create conversation
                conversation = get_active_conversation(db, user.id)
                
                # Save user message (embedding optional for crisis)
                try:
                    user_embedding = self.rag.create_embedding(user_message)
                except Exception as e:
                    logger.error(f"❌ Failed to create embedding for crisis message: {e}")
                    user_embedding = None
                
                save_message(
                    db, conversation.id, user.id, "user", 
                    user_message, user_embedding, contains_crisis=True
                )
                
                # Save crisis response
                try:
                    crisis_embedding = self.rag.create_embedding(self.CRISIS_RESPONSE)
                except Exception:
                    crisis_embedding = None
                
                save_message(
                    db, conversation.id, user.id, "assistant",
                    self.CRISIS_RESPONSE, crisis_embedding
                )
                
                logger.info(f"✅ Sent crisis response to {whatsapp_number}")
                
                return {
                    "response": self.CRISIS_RESPONSE,
                    "is_crisis": True,
                    "user_id": str(user.id)
                }
            
            # Normal therapeutic response flow
            # Get or create conversation
            conversation = get_active_conversation(db, user.id)
            
            # Retrieve conversation history
            try:
                history_messages = get_conversation_history(db, conversation.id, limit=10)
                conversation_history = [
                    {"role": msg.role, "content": msg.content}
                    for msg in history_messages
                ]
            except Exception as e:
                logger.error(f"❌ Error retrieving conversation history: {e}")
                conversation_history = []
            
            # Retrieve relevant context from knowledge base
            try:
                relevant_contexts = self.rag.retrieve_relevant_context(
                    db, user_message, k=5
                )
                logger.info(f"📚 Retrieved {len(relevant_contexts)} context chunks")
            except Exception as e:
                logger.error(f"❌ Error retrieving context: {e}")
                relevant_contexts = []
            
            # Build prompt with context
            try:
                prompt = self.rag.build_prompt_with_context(
                    user_message, relevant_contexts, conversation_history
                )
            except Exception as e:
                logger.error(f"❌ Error building prompt: {e}")
                # Fallback to simple prompt
                prompt = f"You are a compassionate digital wellness therapist. User message: {user_message}"
            
            # Generate response with GPT-4 (with retry logic)
            bot_response = None
            for attempt in range(3):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": prompt},
                            {"role": "user", "content": user_message}
                        ],
                        temperature=0.7,
                        max_tokens=300,
                        presence_penalty=0.6,
                        frequency_penalty=0.3
                    )
                    
                    bot_response = response.choices[0].message.content.strip()
                    break
                
                except RateLimitError as e:
                    wait_time = (attempt + 1) * 2
                    logger.warning(f"⚠️ Rate limit hit, waiting {wait_time}s (attempt {attempt + 1}/3)")
                    time.sleep(wait_time)
                
                except APIError as e:
                    logger.error(f"❌ OpenAI API error: {e}")
                    if attempt < 2:
                        time.sleep(1)
                
                except Exception as e:
                    logger.error(f"❌ Error calling GPT-4: {e}")
                    break
            
            # If GPT-4 failed, use fallback response
            if not bot_response:
                logger.warning("⚠️ GPT-4 failed, using fallback response")
                bot_response = self._get_fallback_response(user_message)
            
            # Save user message
            try:
                user_embedding = self.rag.create_embedding(user_message)
            except Exception as e:
                logger.error(f"❌ Failed to create user message embedding: {e}")
                user_embedding = None
            
            save_message(
                db, conversation.id, user.id, "user",
                user_message, user_embedding
            )
            
            # Save bot response
            try:
                bot_embedding = self.rag.create_embedding(bot_response)
            except Exception as e:
                logger.error(f"❌ Failed to create bot response embedding: {e}")
                bot_embedding = None
            
            save_message(
                db, conversation.id, user.id, "assistant",
                bot_response, bot_embedding
            )
            
            elapsed_time = time.time() - start_time
            logger.info(f"✅ Generated response for {whatsapp_number} in {elapsed_time:.2f}s")
            
            return {
                "response": bot_response,
                "is_crisis": False,
                "user_id": str(user.id)
            }
        
        except Exception as e:
            logger.error(f"❌ Critical error generating response: {e}", exc_info=True)
            
            # Ultimate fallback response
            fallback_response = """I apologize, but I'm having trouble processing your message right now. This is a temporary technical issue on my end.

Please try again in a moment. If you're experiencing a mental health crisis, please contact:
- 988 Suicide & Crisis Lifeline
- Crisis Text Line: Text HOME to 741741

I'm here to help with digital wellness when you're ready to try again."""
            
            return {
                "response": fallback_response,
                "is_crisis": False,
                "user_id": None
            }
    
    def _get_fallback_response(self, user_message: str) -> str:
        """
        Generate a contextual fallback response when GPT-4 is unavailable.
        
        Args:
            user_message: User's message
        
        Returns:
            Appropriate fallback response
        """
        message_lower = user_message.lower()
        
        # Screen time related
        if any(word in message_lower for word in ["screen", "phone", "social media", "instagram", "tiktok"]):
            return """I understand you're concerned about screen time. While I'm experiencing technical difficulties, here's a quick suggestion: Try the "Four Aces" approach - start with Awareness. Simply notice your phone usage patterns without judgment. What triggers make you reach for your phone? 📱

Can you identify one trigger right now?"""
        
        # Stress/anxiety related
        if any(word in message_lower for word in ["stress", "anxiety", "worried", "anxious"]):
            return """I hear that you're feeling stressed. A simple mindfulness technique: Take three deep breaths. Focus on what you can control right now (the Stoic principle of dichotomy of control). What's one small thing within your control today? 🌿"""
        
        # Addiction/habit related
        if any(word in message_lower for word in ["addicted", "habit", "can't stop"]):
            return """Breaking habits is challenging. Start small with the concept of "Acceptance" - acknowledge the habit without self-judgment. Then, what's one tiny step you could take today to shift this pattern? Remember, progress over perfection. 💪"""
        
        # General support
        return """I'm here to support you with digital wellness. While I'm having technical difficulties right now, I want you to know your concerns are valid. Take a moment for yourself - maybe step outside, take three deep breaths, or do something that brings you joy.

What's one small act of self-care you could do right now? 🌟"""
    
    def format_whatsapp_message(self, text: str) -> str:
        """Format message for WhatsApp (handle markdown, emojis, etc.)."""
        # WhatsApp supports basic markdown
        # Bold: *text* or **text**
        # Italic: _text_
        # Strikethrough: ~text~
        # Monospace: ```text```
        
        # Ensure proper formatting
        formatted = text.strip()
        
        # Add spacing for better readability
        formatted = re.sub(r'\n\n\n+', '\n\n', formatted)
        
        return formatted


# Singleton instance
_chatbot_instance: Optional[TherapeuticChatbot] = None


def get_chatbot() -> TherapeuticChatbot:
    """Get or create chatbot singleton instance."""
    global _chatbot_instance
    
    if _chatbot_instance is None:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key or openai_api_key == "your_openai_api_key_here":
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        _chatbot_instance = TherapeuticChatbot(openai_api_key)
        logger.info("Chatbot instance created")
    
    return _chatbot_instance


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test chatbot initialization
    try:
        chatbot = get_chatbot()
        print("Chatbot initialized successfully!")
    except Exception as e:
        print(f"Error initializing chatbot: {e}")

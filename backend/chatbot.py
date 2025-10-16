"""
Therapeutic chatbot logic with OpenAI GPT-4 integration.
Handles response generation, crisis detection, and conversation management.
"""

import os
import re
from typing import List, Dict, Optional
from openai import OpenAI
from sqlalchemy.orm import Session
from rag_system import TherapeuticRAG
from database import (
    get_or_create_user, 
    get_active_conversation, 
    save_message,
    get_conversation_history
)
import logging

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
        Generate therapeutic response using RAG and GPT-4.
        
        Returns:
            Dict with 'response', 'is_crisis', and 'user_id'
        """
        try:
            # Get or create user
            user = get_or_create_user(db, whatsapp_number)
            
            # Check for crisis content
            is_crisis = self.detect_crisis(user_message)
            
            if is_crisis:
                logger.warning(f"Crisis content detected from {whatsapp_number}")
                user.crisis_flag = True
                db.commit()
                
                # Get or create conversation
                conversation = get_active_conversation(db, user.id)
                
                # Save user message
                user_embedding = self.rag.create_embedding(user_message)
                save_message(
                    db, conversation.id, user.id, "user", 
                    user_message, user_embedding, contains_crisis=True
                )
                
                # Save crisis response
                crisis_embedding = self.rag.create_embedding(self.CRISIS_RESPONSE)
                save_message(
                    db, conversation.id, user.id, "assistant",
                    self.CRISIS_RESPONSE, crisis_embedding
                )
                
                return {
                    "response": self.CRISIS_RESPONSE,
                    "is_crisis": True,
                    "user_id": str(user.id)
                }
            
            # Normal therapeutic response flow
            # Get or create conversation
            conversation = get_active_conversation(db, user.id)
            
            # Retrieve conversation history
            history_messages = get_conversation_history(db, conversation.id, limit=10)
            conversation_history = [
                {"role": msg.role, "content": msg.content}
                for msg in history_messages
            ]
            
            # Retrieve relevant context from knowledge base
            relevant_contexts = self.rag.retrieve_relevant_context(
                db, user_message, k=5
            )
            
            # Build prompt with context
            prompt = self.rag.build_prompt_with_context(
                user_message, relevant_contexts, conversation_history
            )
            
            # Generate response with GPT-4
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_tokens=300,  # Keep responses concise (2-3 sentences)
                presence_penalty=0.6,
                frequency_penalty=0.3
            )
            
            bot_response = response.choices[0].message.content.strip()
            
            # Save user message
            user_embedding = self.rag.create_embedding(user_message)
            save_message(
                db, conversation.id, user.id, "user",
                user_message, user_embedding
            )
            
            # Save bot response
            bot_embedding = self.rag.create_embedding(bot_response)
            save_message(
                db, conversation.id, user.id, "assistant",
                bot_response, bot_embedding
            )
            
            logger.info(f"Generated response for {whatsapp_number}")
            
            return {
                "response": bot_response,
                "is_crisis": False,
                "user_id": str(user.id)
            }
        
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            
            # Fallback response
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

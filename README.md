# WhatsApp Therapeutic Chatbot for Digital Wellness 🧠💬

A production-ready WhatsApp chatbot providing therapeutic support for digital wellness, screen time addiction, and technology-related mental health concerns. Built with FastAPI, PostgreSQL + pgvector, LangChain, and OpenAI GPT-4.

## 🎯 Features

- **WhatsApp Integration**: Twilio WhatsApp Business API
- **Therapeutic AI**: GPT-4-powered responses with evidence-based techniques
- **RAG System**: Retrieval-Augmented Generation using PDF knowledge base
- **Vector Search**: PostgreSQL with pgvector for context-aware responses
- **Conversation History**: Persistent storage with embeddings
- **Crisis Detection**: Automatic intervention for crisis keywords

## 🚀 Quick Start

### 1. Configure Environment (.env)

```bash
DATABASE_URL="postgresql://chatbot_user:chatbot_pass@postgres:5432/therapy_chatbot"
TWILIO_ACCOUNT_SID="your_twilio_account_sid"
TWILIO_AUTH_TOKEN="your_twilio_auth_token"
TWILIO_WHATSAPP_NUMBER="whatsapp:+14155238886"
OPENAI_API_KEY="sk-your-openai-api-key"
```

### 2. Start with Docker

```bash
docker-compose up -d
docker-compose logs -f backend
curl http://localhost:8001/api/health
```

### 3. Test the Chatbot

```bash
curl -X POST "http://localhost:8001/api/test-message?message=I%20need%20help%20with%20screen%20time&whatsapp_number=whatsapp:+1234567890"
```

## 📱 Twilio Setup

1. Go to [Twilio Console](https://console.twilio.com/)
2. Set up WhatsApp sandbox or Business API
3. Configure webhook: `https://your-domain.com/api/whatsapp` (POST)

## 🛠️ API Endpoints

- `GET /api/health` - Health check
- `GET /api/status` - System status and statistics  
- `POST /api/whatsapp` - Twilio webhook
- `POST /api/test-message` - Test endpoint

## 📚 Knowledge Base

The chatbot uses two comprehensive therapeutic books by Christian Dominique:

1. **4As_Manuscript_v6.pdf (1.6MB)** - "The Four Aces: Awakening to Happiness"
   - The Four Aces: Awareness, Acceptance, Appreciation, Awe
   - Holistic approach to happiness and well-being
   - Mindfulness, cognitive reframing, positive psychology

2. **BeyondHappy_MANUSCRIPT_v7.pdf (3.5MB)** - "Beyond Happy: Formulas for Perfect Days"
   - The 7Cs: Contentment, Curiosity, Creativity, Compassion, Compersion, Courage, Connection
   - The 8Ps: Presence, Positivity, Purpose, Peace, Playfulness, Passion, Patience, Perseverance
   - Philosophy (Stoicism, Buddhism, Daoism), psychology, neuroscience
   - Internal locus of control and mindset mastery

## 🚨 Crisis Support

Auto-detects crisis keywords and provides:
---

**Built with ❤️ for digital wellness**

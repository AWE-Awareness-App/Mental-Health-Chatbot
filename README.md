# WhatsApp Therapeutic Chatbot for Digital Wellness 🧠💬

A **100% production-ready MVP** WhatsApp chatbot providing therapeutic support for digital wellness, screen time addiction, and technology-related mental health concerns. Built with FastAPI, PostgreSQL + pgvector, LangChain, and OpenAI GPT-4.

## 🚀 **Production MVP - Fully Upgraded**

This chatbot is production-ready with enterprise-grade features:
- ✅ **Persistent pgvector embeddings** (survive Docker restarts)
- ✅ **Source citations** in responses ([Book Title, p.XX])
- ✅ **Robust error handling** with graceful fallbacks
- ✅ **Environment validation** at startup
- ✅ **Rate limiting** (10 msg/min per user)
- ✅ **Enhanced logging** with metrics
- ✅ **Smart initialization** (instant restart)

## 🎯 Core Features

- **WhatsApp Integration**: Twilio WhatsApp Business API with rate limiting
- **Therapeutic AI**: GPT-4-powered responses with evidence-based techniques
- **RAG System with Citations**: Retrieval-Augmented Generation using PDF knowledge base
- **Vector Search**: PostgreSQL with pgvector for persistent, context-aware responses
- **Conversation History**: Full message history with embeddings
- **Crisis Detection**: Automatic intervention for crisis keywords with resource provision
- **Error Handling**: Never crashes - graceful degradation when services fail

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

---

**Built for digital wellness**

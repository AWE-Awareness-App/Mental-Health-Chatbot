# Quick Start Guide - WhatsApp Therapeutic Chatbot

## ⚡ 5-Minute Setup

### 1. Get Your API Keys

**Twilio** (https://console.twilio.com/):
- Sign up for free trial
- Get Account SID, Auth Token
- Enable WhatsApp sandbox

**OpenAI** (https://platform.openai.com/):
- Create account
- Generate API key
- Add billing (needed for GPT-4)

### 2. Configure Environment

```bash
cd /app
cp backend/.env.example backend/.env
```

Edit `backend/.env`:
```bash
TWILIO_ACCOUNT_SID="your_actual_sid"
TWILIO_AUTH_TOKEN="your_actual_token"
TWILIO_WHATSAPP_NUMBER="whatsapp:+14155238886"  # Twilio sandbox number
OPENAI_API_KEY="sk-proj-your_actual_key"
```

### 3. Start with Docker

```bash
docker-compose up -d
docker-compose logs -f backend
```

Wait for: `"Chatbot ready!"`

### 4. Test It

```bash
# Health check
curl http://localhost:8001/api/health

# Test message
curl -X POST "http://localhost:8001/api/test-message?message=I%20spend%20too%20much%20time%20on%20my%20phone&whatsapp_number=whatsapp:+1234567890"
```

### 5. Connect Twilio

**For Local Testing:**
```bash
# Install ngrok
brew install ngrok  # or download from ngrok.com

# Start tunnel
ngrok http 8001

# Copy HTTPS URL: https://abc123.ngrok.io
```

**Configure Webhook:**
1. Go to Twilio Console > WhatsApp Sandbox
2. Set webhook: `https://abc123.ngrok.io/api/whatsapp`
3. Save

### 6. Send WhatsApp Message

1. Join sandbox: Send code to Twilio number
2. Send: "I need help with screen time"
3. Receive therapeutic response! 🎉

## 📱 Try These Messages

**General Support:**
- "I spend too much time on social media"
- "How can I reduce my screen time?"
- "I'm addicted to my phone"

**Crisis Test:**
- "I feel suicidal" → Should get crisis resources

**Therapeutic Techniques:**
- "What is mindfulness?"
- "How do I set boundaries with technology?"
- "Tell me about CBT for phone addiction"

## 🔍 Verify Everything Works

```bash
# Check status
curl http://localhost:8001/api/status

# Expected response:
{
  "status": "operational",
  "database": "connected",
  "statistics": {
    "total_users": X,
    "total_messages": Y,
    "knowledge_base_documents": ~150
  }
}
```

## 🛠️ Common Issues

**"Chatbot not configured"**
- Check OPENAI_API_KEY in .env
- Restart: `docker-compose restart backend`

**"Database connection failed"**
- Wait 30 seconds for PostgreSQL
- Check: `docker-compose ps`

**"Twilio webhook timeout"**
- Ensure ngrok is running
- Check URL in Twilio console
- Verify server responding: `curl http://localhost:8001/api/health`

**"No response from chatbot"**
- Check logs: `docker-compose logs -f backend`
- Test API directly: `/api/test-message`
- Verify OpenAI credits

## 📊 Monitor Your Chatbot

```bash
# Real-time logs
docker-compose logs -f backend

# Database stats
docker-compose exec postgres psql -U chatbot_user therapy_chatbot -c "SELECT COUNT(*) FROM users;"

# Check OpenAI usage
# Go to https://platform.openai.com/usage
```

## 🚀 Next Steps

1. **Production Deployment**: See `DEPLOYMENT.md`
2. **Add Custom Knowledge**: Add PDFs to `backend/knowledge_base/`
3. **Customize Responses**: Edit prompts in `chatbot.py`
4. **Add Features**: Extend API in `server.py`

## 💡 Pro Tips

- **Test Before Production**: Use sandbox thoroughly
- **Monitor Costs**: OpenAI GPT-4 can be expensive
- **Review Conversations**: Check logs weekly
- **Update Knowledge Base**: Add relevant content regularly
- **Crisis Protocols**: Have emergency contact procedures

## 📞 Need Help?

1. Check logs: `docker-compose logs backend`
2. Test system: `python backend/test_system.py`
3. Review: `README.md` and `DEPLOYMENT.md`

---

**You're ready to help people with digital wellness! 🧠✨**

# Deployment Guide for WhatsApp Therapeutic Chatbot

## 📋 Pre-Deployment Checklist

### Required Services and Keys

- [ ] **Twilio Account** - [Sign up](https://www.twilio.com/try-twilio)
  - Account SID
  - Auth Token  
  - WhatsApp enabled number
  
- [ ] **OpenAI Account** - [Get API Key](https://platform.openai.com/api-keys)
  - API key with GPT-4 access
  - Sufficient credits/billing configured

- [ ] **Server/Hosting** - One of:
  - Docker-capable server (recommended)
  - Cloud provider (AWS, GCP, Azure, DigitalOcean)
  - VPS with Python 3.11+
  
- [ ] **Domain Name** (for production)
  - HTTPS required for Twilio webhooks
  - SSL certificate configured

## 🚀 Deployment Options

### Option 1: Docker Deployment (Recommended)

#### 1. Prepare Server

```bash
# Install Docker and Docker Compose
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

#### 2. Clone and Configure

```bash
# Upload project files to server
scp -r /app user@your-server:/path/to/deployment

# SSH into server
ssh user@your-server
cd /path/to/deployment

# Configure environment
cp backend/.env.example backend/.env
nano backend/.env  # Add your API keys
```

#### 3. Configure Environment Variables

Edit `backend/.env`:

```bash
DATABASE_URL="postgresql://chatbot_user:chatbot_pass@postgres:5432/therapy_chatbot"
TWILIO_ACCOUNT_SID="ACxxxxxxxxxxxxxxxxxxxxx"
TWILIO_AUTH_TOKEN="your_auth_token"
TWILIO_WHATSAPP_NUMBER="whatsapp:+14155238886"
OPENAI_API_KEY="sk-proj-xxxxxxxxxxxxx"
CORS_ORIGINS="*"
LOG_LEVEL="INFO"
```

#### 4. Deploy

```bash
# Start services
docker-compose up -d

# Check logs
docker-compose logs -f backend

# Verify health
curl http://localhost:8001/api/health
```

#### 5. Configure Nginx Reverse Proxy (Production)

```nginx
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl;
    server_name your-domain.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://localhost:8001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

#### 6. Set Up Twilio Webhook

1. Go to [Twilio Console](https://console.twilio.com/)
2. Navigate to Messaging > Settings > WhatsApp Sandbox Settings
3. Set "When a message comes in" to: `https://your-domain.com/api/whatsapp`
4. Method: `HTTP POST`
5. Save configuration

### Option 2: Manual Deployment (Without Docker)

#### 1. Install Prerequisites

```bash
# Install PostgreSQL with pgvector
sudo apt-get update
sudo apt-get install postgresql-14 postgresql-14-pgvector python3.11 python3.11-venv

# Create database
sudo -u postgres createdb therapy_chatbot
sudo -u postgres psql therapy_chatbot -c "CREATE EXTENSION vector;"
```

#### 2. Set Up Application

```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
cd backend
pip install -r requirements.txt

# Configure environment
cp .env.example .env
nano .env  # Add your API keys

# Update DATABASE_URL in .env
DATABASE_URL="postgresql://postgres:password@localhost:5432/therapy_chatbot"
```

#### 3. Initialize Database

```bash
python database.py
python rag_system.py  # Index knowledge base
```

#### 4. Run with Systemd

Create `/etc/systemd/system/chatbot.service`:

```ini
[Unit]
Description=WhatsApp Therapeutic Chatbot
After=network.target postgresql.service

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/app/backend
Environment="PATH=/path/to/venv/bin"
ExecStart=/path/to/venv/bin/uvicorn server:app --host 0.0.0.0 --port 8001
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable chatbot
sudo systemctl start chatbot
sudo systemctl status chatbot
```

## 🔍 Post-Deployment Verification

### 1. Health Checks

```bash
# API health
curl https://your-domain.com/api/health

# System status
curl https://your-domain.com/api/status

# Test message (with OpenAI)
curl -X POST "https://your-domain.com/api/test-message?message=Hello&whatsapp_number=whatsapp:+1234567890"
```

### 2. Twilio Webhook Test

1. Send a WhatsApp message to your Twilio number
2. Check logs: `docker-compose logs -f backend` or `journalctl -u chatbot -f`
3. Verify response received on WhatsApp

### 3. Crisis Detection Test

Send message with crisis keyword (e.g., "feeling suicidal") and verify:
- Bot responds with crisis resources
- User flagged in database: `docker-compose exec postgres psql -U chatbot_user therapy_chatbot -c "SELECT * FROM users WHERE crisis_flag = true;"`

### 4. Knowledge Base Verification

```bash
# Check indexed documents
curl https://your-domain.com/api/status | jq '.statistics.knowledge_base_documents'

# Should show ~150 documents from 3 PDFs
```

## 📊 Monitoring

### Application Logs

```bash
# Docker
docker-compose logs -f backend

# Systemd
journalctl -u chatbot -f

# Log file
tail -f backend/logs/chatbot.log
```

### Database Queries

```bash
# Connect to database
docker-compose exec postgres psql -U chatbot_user therapy_chatbot

# Or locally
psql therapy_chatbot

# Useful queries
SELECT COUNT(*) FROM users;
SELECT COUNT(*) FROM messages;
SELECT COUNT(*) FROM knowledge_documents;
SELECT * FROM users WHERE crisis_flag = true;
SELECT COUNT(*) FROM messages WHERE contains_crisis_keywords = true;
```

### Key Metrics

- **Response Time**: Should be < 3 seconds
- **Error Rate**: Should be < 1%
- **Crisis Detections**: Monitor frequency
- **API Costs**: Track OpenAI usage

## 🔧 Maintenance

### Daily

- [ ] Check error logs
- [ ] Monitor API costs
- [ ] Verify service uptime

### Weekly

- [ ] Review crisis interventions
- [ ] Check database size
- [ ] Analyze common user queries

### Monthly

- [ ] Backup database
- [ ] Update dependencies
- [ ] Review and update knowledge base
- [ ] Rotate API keys

### Backup Database

```bash
# Docker
docker-compose exec postgres pg_dump -U chatbot_user therapy_chatbot > backup_$(date +%Y%m%d).sql

# Restore
cat backup_20250101.sql | docker-compose exec -T postgres psql -U chatbot_user therapy_chatbot
```

## 🚨 Troubleshooting

### Service Won't Start

```bash
# Check logs
docker-compose logs backend

# Common issues:
# - Database not ready: wait 30 seconds and retry
# - Port conflict: change port in docker-compose.yml
# - Missing .env: copy from .env.example
```

### Twilio Webhook Errors

```bash
# Check Twilio console for error details
# Common issues:
# - Not HTTPS: configure SSL
# - Wrong URL: verify /api/whatsapp path
# - Timeout: check server performance
```

### OpenAI API Errors

```bash
# Rate limit exceeded: upgrade plan or add delays
# Invalid API key: check .env configuration
# Insufficient credits: add billing
```

### Database Connection Issues

```bash
# Verify PostgreSQL running
docker-compose ps postgres

# Check connection string
docker-compose exec backend printenv DATABASE_URL

# Test connection
docker-compose exec postgres psql -U chatbot_user therapy_chatbot -c "SELECT 1;"
```

## 📈 Scaling

### Horizontal Scaling

```yaml
# docker-compose.yml
services:
  backend:
    deploy:
      replicas: 3
    # Load balancer needed
```

### Database Optimization

```sql
-- Add indexes for common queries
CREATE INDEX idx_messages_conversation ON messages(conversation_id);
CREATE INDEX idx_messages_timestamp ON messages(timestamp DESC);
CREATE INDEX idx_users_whatsapp ON users(whatsapp_number);

-- Vacuum database periodically
VACUUM ANALYZE;
```

### Cost Optimization

- Use GPT-4-turbo instead of GPT-4 (cheaper)
- Cache common responses
- Implement rate limiting per user
- Use smaller embeddings if possible

## 🔐 Security Hardening

### Production Checklist

- [ ] Enable Twilio request validation (uncomment in server.py)
- [ ] Set strong database passwords
- [ ] Use secrets management (AWS Secrets Manager, etc.)
- [ ] Enable HTTPS only
- [ ] Implement rate limiting
- [ ] Set up firewall rules
- [ ] Regular security audits
- [ ] Monitor for suspicious activity

### Environment Secrets

Never commit:
- `.env` files
- API keys
- Database passwords
- SSL certificates

Use environment variables or secrets management:
- AWS Secrets Manager
- HashiCorp Vault
- Docker secrets
- Kubernetes secrets

## 📝 Legal and Compliance

### HIPAA Compliance (if handling health data)

- [ ] Business Associate Agreement with Twilio
- [ ] Encrypted database storage
- [ ] Audit logging
- [ ] Access controls
- [ ] Data retention policies

### Terms of Service

Make clear:
- This is not emergency services
- Not a replacement for professional therapy
- Crisis resources available
- Data privacy policy

## 🎯 Success Metrics

Track:
- User engagement rate
- Average conversation length
- Crisis intervention frequency
- User satisfaction (surveys)
- Response quality (manual review)
- Technical uptime (>99.9%)

---

**Questions or issues?** Check logs first, then consult the troubleshooting section.

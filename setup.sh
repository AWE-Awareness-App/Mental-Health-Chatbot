#!/bin/bash

echo "================================================"
echo "WhatsApp Therapeutic Chatbot Setup"
echo "================================================"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

echo "✅ Docker and Docker Compose are installed"
echo ""

# Check if .env file exists
if [ ! -f "backend/.env" ]; then
    echo "⚠️  .env file not found. Creating from example..."
    cp backend/.env.example backend/.env
    echo "📝 Please edit backend/.env with your API keys and credentials"
    echo ""
    read -p "Press Enter after you've configured backend/.env..."
fi

echo "🔧 Building Docker containers..."
docker-compose build

echo ""
echo "🚀 Starting services..."
docker-compose up -d

echo ""
echo "⏳ Waiting for services to be ready..."
sleep 10

echo ""
echo "🔍 Checking service health..."
docker-compose ps

echo ""
echo "📊 Checking API status..."
curl -s http://localhost:8001/api/health | python -m json.tool || echo "Service not ready yet"

echo ""
echo "================================================"
echo "Setup Complete!"
echo "================================================"
echo ""
echo "📋 Next Steps:"
echo ""
echo "1. Configure Twilio webhook:"
echo "   - Go to https://console.twilio.com/"
echo "   - Set webhook URL to: https://your-domain.com/api/whatsapp"
echo ""
echo "2. Test the chatbot:"
echo "   curl -X POST \"http://localhost:8001/api/test-message?message=Hello&whatsapp_number=whatsapp:+1234567890\""
echo ""
echo "3. View logs:"
echo "   docker-compose logs -f backend"
echo ""
echo "4. Check status:"
echo "   curl http://localhost:8001/api/status"
echo ""
echo "================================================"

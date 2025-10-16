#!/bin/bash

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

BASE_URL="http://localhost:8001"

echo "================================================"
echo "WhatsApp Therapeutic Chatbot - API Tests"
echo "================================================"
echo ""

# Test 1: Health Check
echo -e "${YELLOW}Test 1: Health Check${NC}"
response=$(curl -s "${BASE_URL}/api/health")
if echo "$response" | grep -q "healthy"; then
    echo -e "${GREEN}✅ PASS${NC}: Health check successful"
    echo "$response" | python3 -m json.tool
else
    echo -e "${RED}❌ FAIL${NC}: Health check failed"
    echo "$response"
fi
echo ""

# Test 2: Root Endpoint
echo -e "${YELLOW}Test 2: Root Endpoint${NC}"
response=$(curl -s "${BASE_URL}/api/")
if echo "$response" | grep -q "WhatsApp Therapeutic Chatbot API"; then
    echo -e "${GREEN}✅ PASS${NC}: Root endpoint working"
    echo "$response" | python3 -m json.tool
else
    echo -e "${RED}❌ FAIL${NC}: Root endpoint failed"
    echo "$response"
fi
echo ""

# Test 3: Status Endpoint
echo -e "${YELLOW}Test 3: Status Endpoint${NC}"
response=$(curl -s "${BASE_URL}/api/status")
if echo "$response" | grep -q "status"; then
    echo -e "${GREEN}✅ PASS${NC}: Status endpoint working"
    echo "$response" | python3 -m json.tool
else
    echo -e "${RED}❌ FAIL${NC}: Status endpoint failed"
    echo "$response"
fi
echo ""

# Test 4: Test Message (requires OpenAI key)
echo -e "${YELLOW}Test 4: Test Message Endpoint${NC}"
response=$(curl -s -X POST "${BASE_URL}/api/test-message?message=I%20need%20help%20with%20screen%20time&whatsapp_number=whatsapp:+1234567890")

if echo "$response" | grep -q "success"; then
    echo -e "${GREEN}✅ PASS${NC}: Test message endpoint working"
    echo "$response" | python3 -m json.tool
elif echo "$response" | grep -q "not configured"; then
    echo -e "${YELLOW}⚠️  SKIP${NC}: OpenAI API key not configured"
    echo "$response" | python3 -m json.tool
else
    echo -e "${RED}❌ FAIL${NC}: Test message failed"
    echo "$response"
fi
echo ""

# Test 5: Crisis Detection Test
echo -e "${YELLOW}Test 5: Crisis Detection${NC}"
response=$(curl -s -X POST "${BASE_URL}/api/test-message?message=I%20feel%20suicidal&whatsapp_number=whatsapp:+1234567890")

if echo "$response" | grep -q "is_crisis.*true"; then
    echo -e "${GREEN}✅ PASS${NC}: Crisis detection working"
    echo "$response" | python3 -m json.tool
elif echo "$response" | grep -q "not configured"; then
    echo -e "${YELLOW}⚠️  SKIP${NC}: OpenAI API key not configured"
    echo "$response" | python3 -m json.tool
else
    echo -e "${RED}❌ FAIL${NC}: Crisis detection test failed"
    echo "$response"
fi
echo ""

echo "================================================"
echo "Test Summary"
echo "================================================"
echo ""
echo "Note: Some tests may be skipped if API keys are not configured."
echo "To run full tests, configure backend/.env with:"
echo "  - OPENAI_API_KEY"
echo "  - TWILIO credentials (for webhook tests)"
echo ""

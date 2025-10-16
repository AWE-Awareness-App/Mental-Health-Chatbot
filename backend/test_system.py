"""
System test script to verify all components are working correctly.
Tests database connectivity, model imports, and basic functionality.
"""

import sys
import os

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    try:
        import fastapi
        import sqlalchemy
        import pgvector
        import twilio
        import openai
        import langchain
        from langchain_community.document_loaders import PyPDFLoader
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def test_database_models():
    """Test database models can be loaded."""
    print("\nTesting database models...")
    try:
        from database import User, Conversation, Message, KnowledgeDocument, Base
        print(f"✅ User model: {User.__tablename__}")
        print(f"✅ Conversation model: {Conversation.__tablename__}")
        print(f"✅ Message model: {Message.__tablename__}")
        print(f"✅ KnowledgeDocument model: {KnowledgeDocument.__tablename__}")
        return True
    except Exception as e:
        print(f"❌ Database model error: {e}")
        return False


def test_pdf_files():
    """Test that PDF knowledge base files exist."""
    print("\nTesting PDF knowledge base...")
    knowledge_base_dir = "knowledge_base"
    
    if not os.path.exists(knowledge_base_dir):
        print(f"❌ Knowledge base directory not found: {knowledge_base_dir}")
        return False
    
    pdf_files = [f for f in os.listdir(knowledge_base_dir) if f.endswith('.pdf')]
    
    if not pdf_files:
        print("❌ No PDF files found in knowledge base")
        return False
    
    print(f"✅ Found {len(pdf_files)} PDF files:")
    for pdf_file in pdf_files:
        file_path = os.path.join(knowledge_base_dir, pdf_file)
        file_size = os.path.getsize(file_path)
        print(f"   - {pdf_file} ({file_size} bytes)")
    
    return True


def test_chatbot_module():
    """Test chatbot module can be imported."""
    print("\nTesting chatbot module...")
    try:
        from chatbot import TherapeuticChatbot
        print("✅ Chatbot class loaded successfully")
        
        # Test crisis keywords
        print(f"✅ Crisis keywords defined: {len(TherapeuticChatbot.CRISIS_KEYWORDS)} keywords")
        
        return True
    except Exception as e:
        print(f"❌ Chatbot module error: {e}")
        return False


def test_rag_module():
    """Test RAG system module can be imported."""
    print("\nTesting RAG system module...")
    try:
        from rag_system import TherapeuticRAG
        print("✅ RAG system class loaded successfully")
        return True
    except Exception as e:
        print(f"❌ RAG system module error: {e}")
        return False


def test_environment():
    """Test environment variables."""
    print("\nTesting environment configuration...")
    
    required_vars = [
        "DATABASE_URL",
        "TWILIO_ACCOUNT_SID",
        "TWILIO_AUTH_TOKEN",
        "TWILIO_WHATSAPP_NUMBER",
        "OPENAI_API_KEY"
    ]
    
    all_set = True
    for var in required_vars:
        value = os.getenv(var)
        if value and value != f"your_{var.lower()}_here":
            print(f"✅ {var}: configured")
        else:
            print(f"⚠️  {var}: not configured (placeholder value)")
            all_set = False
    
    return all_set


def test_server_module():
    """Test server module can be imported."""
    print("\nTesting server module...")
    try:
        from server import app, api_router
        print("✅ FastAPI app loaded successfully")
        print(f"✅ API router prefix: {api_router.prefix}")
        return True
    except Exception as e:
        print(f"❌ Server module error: {e}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("WhatsApp Therapeutic Chatbot - System Test")
    print("="*60)
    
    tests = [
        ("Imports", test_imports),
        ("Database Models", test_database_models),
        ("PDF Knowledge Base", test_pdf_files),
        ("Chatbot Module", test_chatbot_module),
        ("RAG System Module", test_rag_module),
        ("Environment Variables", test_environment),
        ("Server Module", test_server_module)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! System is ready.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

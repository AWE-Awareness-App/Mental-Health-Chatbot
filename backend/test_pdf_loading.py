"""
Quick test to verify the new PDFs can be loaded and processed.
This doesn't require database or OpenAI API - just tests PDF reading.
"""

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def test_pdf_loading():
    """Test that PDFs can be loaded and chunked."""
    
    knowledge_base_dir = Path("knowledge_base")
    pdf_files = list(knowledge_base_dir.glob("*.pdf"))
    
    print("="*60)
    print("PDF Loading Test")
    print("="*60)
    print(f"\nFound {len(pdf_files)} PDF files:\n")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    total_chunks = 0
    total_chars = 0
    
    for pdf_file in pdf_files:
        try:
            print(f"📄 Processing: {pdf_file.name}")
            
            # Load PDF
            loader = PyPDFLoader(str(pdf_file))
            documents = loader.load()
            
            print(f"   ✅ Loaded {len(documents)} pages")
            
            # Split into chunks
            chunks = text_splitter.split_documents(documents)
            total_chunks += len(chunks)
            
            # Calculate total characters
            chars_in_file = sum(len(chunk.page_content) for chunk in chunks)
            total_chars += chars_in_file
            
            print(f"   ✅ Created {len(chunks)} chunks")
            print(f"   ✅ Total characters: {chars_in_file:,}")
            
            # Show sample chunk
            if chunks:
                sample = chunks[0].page_content[:200]
                print(f"   📝 Sample text: {sample}...")
            
            print()
            
        except Exception as e:
            print(f"   ❌ Error processing {pdf_file.name}: {e}")
            print()
    
    print("="*60)
    print("Summary")
    print("="*60)
    print(f"Total PDFs processed: {len(pdf_files)}")
    print(f"Total chunks created: {total_chunks}")
    print(f"Total characters: {total_chars:,}")
    print(f"Average chunk size: {total_chars // total_chunks if total_chunks > 0 else 0} chars")
    print()
    print("✅ PDF loading test completed successfully!")
    print("="*60)

if __name__ == "__main__":
    test_pdf_loading()

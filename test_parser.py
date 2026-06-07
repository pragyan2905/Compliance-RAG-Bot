import os
import fitz
from core.document_parser import DocumentParser

def create_test_pdf(filename: str):
    doc = fitz.open()
    page = doc.new_page()
    
    # Major Heading
    page.insert_text((50, 50), "ARTICLE 1: CONFIDENTIALITY", fontsize=16, fontname="helv", color=(0,0,0))
    # Subsection
    page.insert_text((50, 80), "1.1 Definition of Confidential Information", fontsize=12, fontname="helv", color=(0,0,0))
    
    # Body text
    y = 110
    for i in range(15):
        page.insert_text((50, y), f"Line {i}: The term 'Confidential Information' means any information or data disclosed.", fontsize=10, fontname="helv")
        y += 15
        
    # Another section
    page.insert_text((50, y+30), "ARTICLE 2: TERM AND TERMINATION", fontsize=16, fontname="helv", color=(0,0,0))
    # Clause
    page.insert_text((50, y+60), "Clause 2.1 This agreement shall commence on the Effective Date.", fontsize=10, fontname="helv")
    
    doc.save(filename)
    doc.close()

def main():
    test_pdf = "test_contract.pdf"
    print("Creating test PDF...")
    create_test_pdf(test_pdf)
    
    print("Parsing test PDF...")
    parser = DocumentParser(chunk_size=200, chunk_overlap=50) # Small chunk size for testing
    parsed_doc = parser.parse(test_pdf)
    
    print(f"\nParsed Document ID: {parsed_doc.document_id}")
    print(f"Filename: {parsed_doc.filename}")
    print(f"Total Chunks: {len(parsed_doc.chunks)}\n")
    
    for i, chunk in enumerate(parsed_doc.chunks):
        print(f"--- Chunk {i+1} ---")
        print(f"Page: {chunk.page}")
        print(f"Section: {chunk.section}")
        print(f"Subsection: {chunk.subsection}")
        print(f"Clause ID: {chunk.clause_id}")
        print(f"Text length: {len(chunk.text)}")
        print(f"Text preview: {chunk.text[:100].replace(chr(10), ' ')}...\n")
        
    # Cleanup
    if os.path.exists(test_pdf):
        os.remove(test_pdf)

if __name__ == "__main__":
    main()

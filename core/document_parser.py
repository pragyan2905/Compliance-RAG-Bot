import re
import uuid
from typing import List, Optional, Tuple, Dict, Any
import fitz  # PyMuPDF

from schemas.document import DocumentChunk, ParsedDocument

class DocumentParser:
    """
    Parses PDF contracts, extracting hierarchical metadata and splitting text into chunks.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Regex to capture things like "1.1", "Section 2(a)", "Clause 3"
        self.clause_pattern = re.compile(r"^(?:Clause|Section)?\s*(\d+(?:\.\d+)*|[A-Z]?[IVX]+|\d+\([a-z]\))\.?\s", re.IGNORECASE)

    def parse(self, filepath: str, document_id: str = None) -> ParsedDocument:
        """
        Parses a PDF file and returns a ParsedDocument object containing metadata-aware chunks.
        """
        if not document_id:
            document_id = str(uuid.uuid4())
            
        doc = fitz.open(filepath)
        filename = filepath.split('/')[-1]
        
        # Heuristics: find base font size to identify headings
        base_font_size = self._estimate_base_font_size(doc)
        
        chunks: List[DocumentChunk] = []
        
        current_section: Optional[str] = None
        current_subsection: Optional[str] = None
        current_clause_id: Optional[str] = None
        
        current_chunk_text = ""
        current_chunk_page = 1
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("dict").get("blocks", [])
            
            for block in blocks:
                if block.get("type") != 0:  # Not a text block
                    continue
                    
                block_text, is_heading, heading_level = self._analyze_block(block, base_font_size)
                
                if not block_text.strip():
                    continue
                    
                # Extract clause ID if present at the start of the block
                clause_match = self.clause_pattern.match(block_text.strip())
                if clause_match:
                    new_clause_id = clause_match.group(1)
                else:
                    new_clause_id = current_clause_id
                
                # Check if we need to save the current chunk before updating metadata
                if len(current_chunk_text) + len(block_text) > self.chunk_size and current_chunk_text:
                    chunks.append(
                        DocumentChunk(
                            chunk_id=str(uuid.uuid4()),
                            text=current_chunk_text.strip(),
                            page=current_chunk_page,
                            section=current_section,
                            subsection=current_subsection,
                            clause_id=current_clause_id
                        )
                    )
                    # Start new chunk with overlap
                    overlap_text = current_chunk_text[-self.chunk_overlap:] if self.chunk_overlap > 0 else ""
                    space_idx = overlap_text.find(' ')
                    if space_idx != -1:
                        overlap_text = overlap_text[space_idx+1:]
                    
                    current_chunk_text = overlap_text + "\n" + block_text
                    current_chunk_page = page_num + 1
                else:
                    current_chunk_text += block_text + "\n"
                    if not current_chunk_text.strip():
                        current_chunk_page = page_num + 1

                # Now update section/subsection for the NEXT chunks if it's a heading
                if is_heading:
                    if heading_level == 1:
                        current_section = block_text.strip()
                        current_subsection = None # Reset subsection
                    else:
                        current_subsection = block_text.strip()
                
                if clause_match:
                    current_clause_id = new_clause_id
        
        # Add the final chunk if any remains
        if current_chunk_text.strip():
            chunks.append(
                DocumentChunk(
                    chunk_id=str(uuid.uuid4()),
                    text=current_chunk_text.strip(),
                    page=current_chunk_page,
                    section=current_section,
                    subsection=current_subsection,
                    clause_id=current_clause_id
                )
            )
            
        doc.close()
        return ParsedDocument(
            document_id=document_id,
            filename=filename,
            chunks=chunks
        )

    def _estimate_base_font_size(self, doc: fitz.Document) -> float:
        """
        Estimates the base font size of the document by sampling the first few pages.
        """
        font_sizes: Dict[float, int] = {}
        for page_num in range(min(3, len(doc))):
            page = doc[page_num]
            blocks = page.get_text("dict").get("blocks", [])
            for block in blocks:
                if block.get("type") == 0:
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            size = round(span.get("size", 0), 1)
                            font_sizes[size] = font_sizes.get(size, 0) + len(span.get("text", ""))
                            
        if not font_sizes:
            return 11.0 # Default assumption
            
        # The font size with the most text is likely the base font size
        return max(font_sizes.items(), key=lambda x: x[1])[0]

    def _analyze_block(self, block: Dict[str, Any], base_font_size: float) -> Tuple[str, bool, int]:
        """
        Analyzes a block to extract its text and determine if it's a heading.
        Returns: (text, is_heading, heading_level)
        """
        text_parts = []
        max_font_size = 0.0
        is_bold = False
        
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text_parts.append(span.get("text", ""))
                size = span.get("size", 0)
                if size > max_font_size:
                    max_font_size = size
                # Check if font name contains "Bold" or flag indicates bold (flag bit 4 is usually bold in PyMuPDF)
                if "bold" in span.get("font", "").lower() or (span.get("flags", 0) & 16):
                    is_bold = True
                    
        text = "".join(text_parts)
        
        # Heuristic for headings
        is_heading = False
        heading_level = 0
        
        # Remove trailing/leading spaces for length check
        clean_text = text.strip()
        
        # Headings are typically short and either bold or larger than base text
        if 0 < len(clean_text) < 150:
            if max_font_size > base_font_size + 2:
                is_heading = True
                heading_level = 1 # Major section
            elif max_font_size > base_font_size + 0.5 or (is_bold and max_font_size >= base_font_size):
                is_heading = True
                heading_level = 2 # Subsection
                
        return text, is_heading, heading_level

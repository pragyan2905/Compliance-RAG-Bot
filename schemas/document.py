from typing import Optional
from pydantic import BaseModel, Field

class DocumentChunk(BaseModel):
    """
    Represents a chunk of text extracted from a legal document.
    Includes hierarchical metadata and traceability identifiers.
    """
    chunk_id: str = Field(description="Unique identifier for the chunk")
    text: str = Field(description="The extracted text content")
    page: int = Field(description="Page number where the chunk begins (1-indexed)")
    section: Optional[str] = Field(default=None, description="The detected section heading")
    subsection: Optional[str] = Field(default=None, description="The detected subsection heading")
    clause_id: Optional[str] = Field(default=None, description="Extracted clause identifier if present (e.g., '1.1')")

class ParsedDocument(BaseModel):
    """
    Represents a fully parsed document containing multiple chunks.
    """
    document_id: str = Field(description="Unique identifier for the document")
    filename: str = Field(description="Original filename")
    chunks: list[DocumentChunk] = Field(default_factory=list, description="Ordered list of text chunks")

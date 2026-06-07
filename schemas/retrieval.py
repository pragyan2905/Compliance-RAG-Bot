from typing import Optional, List
from pydantic import BaseModel, Field

class VectorRecord(BaseModel):
    """
    Represents a chunk and its dense embedding for vector database storage.
    """
    id: str = Field(description="Unique identifier for the vector record (matches chunk_id)")
    vector: List[float] = Field(description="Dense vector embedding of the text chunk")
    text: str = Field(description="The text content of the chunk")
    document_id: str = Field(description="ID of the parent document")
    page: int = Field(description="Page number where the chunk begins")
    section: Optional[str] = Field(default=None, description="The detected section heading")
    subsection: Optional[str] = Field(default=None, description="The detected subsection heading")
    clause_id: Optional[str] = Field(default=None, description="Extracted clause identifier if present")

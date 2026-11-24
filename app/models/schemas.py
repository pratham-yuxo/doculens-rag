from pydantic import BaseModel, Field
from typing import List, Optional

class UploadResponse(BaseModel):
    filename: str
    status: str
    message: str

class DocumentInfo(BaseModel):
    filename: str
    chunks: int
    pages: int

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    question: str
    history: List[Message] = []
    top_k: int = 5
    # Optional: Allow frontend to force a specific mode if needed in future
    use_mmr: bool = True 

class RetrievedChunk(BaseModel):
    score: float
    filename: str
    page_no: int
    chunk_id: int
    text_snippet: str

class ChatAnswer(BaseModel):
    answer: str
    sources: List[RetrievedChunk]
    refused: bool = False
    reasoning: Optional[str] = None
    # NEW: Helps frontend understand if we did a specific search or a broad summary
    intent: str = Field(default="unknown", description="The classification mode used: 'factual' or 'analysis'")
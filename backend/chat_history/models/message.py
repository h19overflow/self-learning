"""
Message data model for chat history system.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime
import json


class Message(BaseModel):
    """
    Message model for chat history system.
    
    Represents a single message (human or AI) within a chat session.
    """
    
    id: Optional[int] = Field(None, description="Unique message identifier")
    session_id: int = Field(..., description="ID of the session this message belongs to")
    message_type: str = Field(..., description="Type of message: 'human' or 'ai'")
    content: str = Field(..., description="Message content", min_length=1)
    sources: Optional[List[str]] = Field(None, description="List of source files for AI responses")
    created_at: Optional[datetime] = Field(None, description="Message creation timestamp")
    
    @validator('message_type')
    def validate_message_type(cls, v):
        """Ensure message_type is either 'human' or 'ai'."""
        if v not in ['human', 'ai']:
            raise ValueError("message_type must be either 'human' or 'ai'")
        return v
    
    class Config:
        """Pydantic configuration."""
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }
    
    def to_dict(self) -> dict:
        """Convert message to dictionary format."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "message_type": self.message_type,
            "content": self.content,
            "sources": json.dumps(self.sources) if self.sources else None,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Message":
        """Create Message instance from dictionary."""
        # Parse sources JSON string back to list
        if data.get("sources") and isinstance(data["sources"], str):
            try:
                data["sources"] = json.loads(data["sources"])
            except json.JSONDecodeError:
                data["sources"] = None
        
        # Parse datetime string
        if data.get("created_at") and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        
        return cls(**data)


class MessageCreate(BaseModel):
    """
    Message creation model without ID and timestamp.
    """
    session_id: int = Field(..., description="Session ID")
    message_type: str = Field(..., description="Message type: 'human' or 'ai'")
    content: str = Field(..., description="Message content", min_length=1)
    sources: Optional[List[str]] = Field(None, description="Source files for AI responses")
    
    @validator('message_type')
    def validate_message_type(cls, v):
        """Ensure message_type is either 'human' or 'ai'."""
        if v not in ['human', 'ai']:
            raise ValueError("message_type must be either 'human' or 'ai'")
        return v


class MessageResponse(BaseModel):
    """
    Message response model with all information.
    """
    id: int = Field(..., description="Message ID")
    session_id: int = Field(..., description="Session ID")
    message_type: str = Field(..., description="Message type")
    content: str = Field(..., description="Message content")
    sources: Optional[List[str]] = Field(None, description="Source files")
    created_at: datetime = Field(..., description="Creation timestamp")
    
    class Config:
        """Pydantic configuration."""
        from_attributes = True
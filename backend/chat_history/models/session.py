"""
Session data model for chat history system.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class Session(BaseModel):
    """
    Session model for chat history system.
    
    Represents a chat session belonging to a user, containing multiple messages.
    """
    
    id: Optional[int] = Field(None, description="Unique session identifier")
    user_id: int = Field(..., description="ID of the user who owns this session")
    session_name: Optional[str] = Field(None, description="Optional name for the session", max_length=100)
    created_at: Optional[datetime] = Field(None, description="Session creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    
    class Config:
        """Pydantic configuration."""
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }
    
    def to_dict(self) -> dict:
        """Convert session to dictionary format."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "session_name": self.session_name,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        """Create Session instance from dictionary."""
        if data.get("created_at") and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if data.get("updated_at") and isinstance(data["updated_at"], str):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        return cls(**data)


class SessionCreate(BaseModel):
    """
    Session creation model without ID and timestamps.
    """
    user_id: int = Field(..., description="User ID")
    session_name: Optional[str] = Field(None, description="Session name", max_length=100)


class SessionResponse(BaseModel):
    """
    Session response model with all information.
    """
    id: int = Field(..., description="Session ID")
    user_id: int = Field(..., description="User ID")
    session_name: Optional[str] = Field(None, description="Session name")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Update timestamp")
    message_count: Optional[int] = Field(None, description="Number of messages in session")
    
    class Config:
        """Pydantic configuration."""
        from_attributes = True
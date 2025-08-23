"""
User data model for chat history system.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class User(BaseModel):
    """
    User model for chat history system.
    
    Represents a user who can have multiple chat sessions.
    """
    
    id: Optional[int] = Field(None, description="Unique user identifier")
    username: str = Field(..., description="Unique username for the user", min_length=3, max_length=50)
    password: str = Field(..., description="User password", min_length=6)
    created_at: Optional[datetime] = Field(None, description="User creation timestamp")
    
    class Config:
        """Pydantic configuration."""
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }
    
    def to_dict(self) -> dict:
        """Convert user to dictionary format."""
        return {
            "id": self.id,
            "username": self.username,
            "password": self.password,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "User":
        """Create User instance from dictionary."""
        if data.get("created_at") and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


class UserCreate(BaseModel):
    """
    User creation model without ID and timestamp.
    """
    username: str = Field(..., description="Unique username", min_length=3, max_length=50)
    password: str = Field(..., description="User password", min_length=6)


class UserResponse(BaseModel):
    """
    User response model without sensitive information.
    """
    id: int = Field(..., description="User ID")
    username: str = Field(..., description="Username")
    created_at: datetime = Field(..., description="Creation timestamp")
    
    class Config:
        """Pydantic configuration."""
        from_attributes = True
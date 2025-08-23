"""
Chat History Manager - Main Interface

This module provides a high-level interface for managing chat history,
combining all CRUD operations into a single, easy-to-use class.
"""

from typing import Optional, Dict, Any, List
from pathlib import Path

from .database.connection import DatabaseConnection
from .crud.user_crud import UserCRUD
from .crud.session_crud import SessionCRUD
from .crud.message_crud import MessageCRUD
from .models.user import User, UserCreate, UserResponse
from .models.session import Session, SessionCreate, SessionResponse
from .models.message import Message, MessageCreate, MessageResponse


class ChatHistoryManager:
    """
    Main interface for chat history management.
    
    This class provides a unified interface for all chat history operations,
    including user management, session handling, and message storage.
    """
    
    def __init__(self, db_path: str = "chat_history.db"):
        """
        Initialize chat history manager with database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db = DatabaseConnection(db_path)
        
        # Initialize CRUD classes
        self.users = UserCRUD(self.db)
        self.sessions = SessionCRUD(self.db)
        self.messages = MessageCRUD(self.db)
    
    # USER MANAGEMENT METHODS
    
    def create_user(self, username: str, password: str) -> int:
        """
        Create a new user.
        
        Args:
            username: Unique username
            password: User password
            
        Returns:
            int: User ID
            
        Raises:
            ValueError: If username already exists
        """
        user_data = {"username": username, "password": password}
        return self.users.create(user_data)
    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """
        Authenticate a user with username and password.
        
        Args:
            username: Username
            password: Password
            
        Returns:
            Optional[Dict[str, Any]]: User data if authentication successful
        """
        return self.users.authenticate(username, password)
    
    def get_user(self, user_id: int) -> Optional[Dict[str, Any]]:
        """
        Get user information by ID.
        
        Args:
            user_id: User ID
            
        Returns:
            Optional[Dict[str, Any]]: User data
        """
        return self.users.get_by_id(user_id)
    
    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """
        Get user information by username.
        
        Args:
            username: Username
            
        Returns:
            Optional[Dict[str, Any]]: User data
        """
        return self.users.get_by_username(username)
    
    def get_all_users(self) -> List[Dict[str, Any]]:
        """
        Get all users.
        
        Returns:
            List[Dict[str, Any]]: List of all users
        """
        return self.users.get_all()
    
    def get_user_sessions(self, user_id: int) -> List[Dict[str, Any]]:
        """
        Get all sessions for a specific user.
        
        Args:
            user_id: User ID
            
        Returns:
            List[Dict[str, Any]]: List of user's sessions
        """
        return self.users.get_user_sessions(user_id)
    
    def get_user_stats(self, user_id: int) -> Dict[str, Any]:
        """
        Get statistics for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Dict[str, Any]: User statistics
        """
        return self.users.get_user_stats(user_id)
    
    # SESSION MANAGEMENT METHODS
    
    def create_session(self, user_id: int, session_name: Optional[str] = None) -> int:
        """
        Create a new chat session.
        
        Args:
            user_id: ID of the user creating the session
            session_name: Optional name for the session
            
        Returns:
            int: Session ID
        """
        session_data = {"user_id": user_id, "session_name": session_name}
        return self.sessions.create(session_data)
    
    def get_session(self, session_id: int) -> Optional[Dict[str, Any]]:
        """
        Get session information by ID.
        
        Args:
            session_id: Session ID
            
        Returns:
            Optional[Dict[str, Any]]: Session data with message count
        """
        return self.sessions.get_by_id(session_id)
    
    def update_session(self, session_id: int, session_name: str) -> bool:
        """
        Update session name.
        
        Args:
            session_id: Session ID
            session_name: New session name
            
        Returns:
            bool: True if update successful
        """
        return self.sessions.update(session_id, {"session_name": session_name})
    
    def delete_session(self, session_id: int) -> bool:
        """
        Delete a session and all its messages.
        
        Args:
            session_id: Session ID
            
        Returns:
            bool: True if deletion successful
        """
        return self.sessions.delete(session_id)
    
    def get_all_sessions(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get all sessions across all users.
        
        Args:
            limit: Maximum number of sessions to return
            
        Returns:
            List[Dict[str, Any]]: List of all sessions
        """
        return self.sessions.get_all(limit=limit)
    
    def get_session_messages(self, session_id: int) -> List[Dict[str, Any]]:
        """
        Get all messages for a specific session.
        
        Args:
            session_id: Session ID
            
        Returns:
            List[Dict[str, Any]]: List of messages in chronological order
        """
        return self.sessions.get_session_messages(session_id)
    
    def get_recent_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recently updated sessions.
        
        Args:
            limit: Maximum number of sessions to return
            
        Returns:
            List[Dict[str, Any]]: List of recent sessions
        """
        return self.sessions.get_recent_sessions(limit)
    
    def get_session_stats(self, session_id: int) -> Dict[str, Any]:
        """
        Get statistics for a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Dict[str, Any]: Session statistics
        """
        return self.sessions.get_session_stats(session_id)
    
    # MESSAGE MANAGEMENT METHODS
    
    def add_human_message(self, session_id: int, content: str) -> int:
        """
        Add a human message to a session.
        
        Args:
            session_id: Session ID
            content: Human message content
            
        Returns:
            int: Message ID
        """
        return self.messages.add_human_message(session_id, content)
    
    def add_ai_message(self, session_id: int, content: str, sources: Optional[List[str]] = None) -> int:
        """
        Add an AI message to a session with optional sources.
        
        Args:
            session_id: Session ID
            content: AI response content
            sources: Optional list of source files
            
        Returns:
            int: Message ID
        """
        return self.messages.add_ai_message(session_id, content, sources)
    
    def add_message(self, session_id: int, message_type: str, content: str, sources: Optional[List[str]] = None) -> int:
        """
        Add a message to a session.
        
        Args:
            session_id: Session ID
            message_type: 'human' or 'ai'
            content: Message content
            sources: Optional list of source files
            
        Returns:
            int: Message ID
        """
        return self.messages.add_message(session_id, message_type, content, sources)
    
    def get_message(self, message_id: int) -> Optional[Dict[str, Any]]:
        """
        Get message by ID.
        
        Args:
            message_id: Message ID
            
        Returns:
            Optional[Dict[str, Any]]: Message data
        """
        return self.messages.get_by_id(message_id)
    
    def update_message(self, message_id: int, content: str, sources: Optional[List[str]] = None) -> bool:
        """
        Update message content and sources.
        
        Args:
            message_id: Message ID
            content: Updated content
            sources: Updated sources
            
        Returns:
            bool: True if update successful
        """
        update_data = {"content": content}
        if sources is not None:
            update_data["sources"] = sources
        return self.messages.update(message_id, update_data)
    
    def delete_message(self, message_id: int) -> bool:
        """
        Delete a message.
        
        Args:
            message_id: Message ID
            
        Returns:
            bool: True if deletion successful
        """
        return self.messages.delete(message_id)
    
    def get_recent_messages(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get recent messages across all sessions.
        
        Args:
            limit: Maximum number of messages to return
            
        Returns:
            List[Dict[str, Any]]: List of recent messages
        """
        return self.messages.get_recent_messages(limit)
    
    def search_messages(self, search_term: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Search messages by content.
        
        Args:
            search_term: Search term
            limit: Maximum number of results
            
        Returns:
            List[Dict[str, Any]]: List of matching messages
        """
        return self.messages.search_messages(search_term, limit)
    
    # CONVERSATION WORKFLOW METHODS
    
    def start_conversation(self, user_id: int, session_name: Optional[str] = None) -> int:
        """
        Start a new conversation session.
        
        Args:
            user_id: User ID
            session_name: Optional session name
            
        Returns:
            int: Session ID
        """
        return self.create_session(user_id, session_name)
    
    def add_conversation_pair(self, session_id: int, human_message: str, ai_response: str, sources: Optional[List[str]] = None) -> Dict[str, int]:
        """
        Add a complete Q&A pair to a session.
        
        Args:
            session_id: Session ID
            human_message: Human question/message
            ai_response: AI response
            sources: Optional list of source files for AI response
            
        Returns:
            Dict[str, int]: Dictionary with human_message_id and ai_message_id
        """
        human_id = self.add_human_message(session_id, human_message)
        ai_id = self.add_ai_message(session_id, ai_response, sources)
        
        return {
            "human_message_id": human_id,
            "ai_message_id": ai_id
        }
    
    def get_session_conversation(self, session_id: int) -> List[Dict[str, Any]]:
        """
        Get complete conversation for a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            List[Dict[str, Any]]: List of messages in conversation order
        """
        return self.get_session_messages(session_id)
    
    # UTILITY METHODS
    
    def get_database_info(self) -> Dict[str, Any]:
        """
        Get database information and statistics.
        
        Returns:
            Dict[str, Any]: Database information
        """
        return self.db.get_database_info()
    
    def close_database(self):
        """Close database connections."""
        self.db.close_all_connections()
    
    def export_user_data(self, user_id: int) -> Dict[str, Any]:
        """
        Export all data for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Dict[str, Any]: Complete user data export
        """
        user = self.get_user(user_id)
        if not user:
            return {"error": "User not found"}
        
        sessions = self.get_user_sessions(user_id)
        
        # Get messages for each session
        for session in sessions:
            session['messages'] = self.get_session_messages(session['id'])
        
        return {
            "user": user,
            "sessions": sessions,
            "stats": self.get_user_stats(user_id)
        }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get system-wide statistics.
        
        Returns:
            Dict[str, Any]: System statistics
        """
        try:
            total_users = self.users.count()
            total_sessions = self.sessions.count()
            total_messages = self.messages.count()
            
            # Get message type breakdown
            human_messages = self.messages.count("message_type = ?", ("human",))
            ai_messages = self.messages.count("message_type = ?", ("ai",))
            
            return {
                "total_users": total_users,
                "total_sessions": total_sessions,
                "total_messages": total_messages,
                "human_messages": human_messages,
                "ai_messages": ai_messages,
                "database_info": self.get_database_info()
            }
            
        except Exception as e:
            return {"error": str(e)}
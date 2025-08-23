"""
Message CRUD operations for chat history system.

This module provides comprehensive message management operations including
specialized methods for adding human and AI messages with source tracking.
"""

import json
from typing import Optional, Dict, Any, List
from .base_crud import BaseCRUD
from backend.chat_history.models.message import Message, MessageCreate


class MessageCRUD(BaseCRUD):
    """
    CRUD operations for messages with specialized add_message functionality.
    """
    
    def _get_table_name(self) -> str:
        """Get table name for messages."""
        return "messages"
    
    def create(self, data: Dict[str, Any]) -> int:
        """
        Create a new message.
        
        Args:
            data: Dictionary containing message data
            
        Returns:
            int: ID of created message
            
        Raises:
            ValueError: If session_id is invalid or message_type is wrong
            Exception: If creation fails
        """
        try:
            # Validate input
            message_create = MessageCreate(**data)
            
            # Verify session exists
            session_exists_query = "SELECT 1 FROM sessions WHERE id = ? LIMIT 1"
            session_result = self.db.execute_query(session_exists_query, (message_create.session_id,))
            if not session_result:
                raise ValueError(f"Session with ID {message_create.session_id} does not exist")
            
            # Prepare message data
            create_data = {
                "session_id": message_create.session_id,
                "message_type": message_create.message_type,
                "content": message_create.content,
                "sources": json.dumps(message_create.sources) if message_create.sources else None
            }
            
            query, params = self._build_insert_query(create_data)
            message_id = self.db.execute_insert(query, params)
            
            self.db.logger.info(f"Message created: {message_create.message_type} message (ID: {message_id})")
            return message_id
            
        except Exception as e:
            self.db.logger.error(f"Failed to create message: {e}")
            raise
    
    def get_by_id(self, record_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve a message by ID with parsed sources.
        
        Args:
            record_id: Message ID
            
        Returns:
            Optional[Dict[str, Any]]: Message data with parsed sources
        """
        try:
            query = """
            SELECT id, session_id, message_type, content, sources, created_at
            FROM messages 
            WHERE id = ?
            """
            
            results = self.db.execute_query(query, (record_id,))
            if not results:
                return None
            
            message_data = results[0]
            
            # Parse sources JSON
            if message_data['sources']:
                try:
                    message_data['sources'] = json.loads(message_data['sources'])
                except json.JSONDecodeError:
                    message_data['sources'] = None
            
            return message_data
            
        except Exception as e:
            self.db.logger.error(f"Failed to get message by ID {record_id}: {e}")
            return None
    
    def update(self, record_id: int, data: Dict[str, Any]) -> bool:
        """
        Update message information.
        
        Args:
            record_id: Message ID
            data: Dictionary containing updated data
            
        Returns:
            bool: True if update successful
        """
        try:
            # Only allow updating content and sources
            allowed_fields = {'content', 'sources'}
            update_data = {k: v for k, v in data.items() if k in allowed_fields}
            
            if not update_data:
                self.db.logger.warning(f"No valid fields to update for message {record_id}")
                return False
            
            # Convert sources list to JSON if present
            if 'sources' in update_data and update_data['sources'] is not None:
                update_data['sources'] = json.dumps(update_data['sources'])
            
            query, params = self._build_update_query(record_id, update_data)
            rows_affected = self.db.execute_update(query, params)
            
            return rows_affected > 0
            
        except Exception as e:
            self.db.logger.error(f"Failed to update message {record_id}: {e}")
            return False
    
    def delete(self, record_id: int) -> bool:
        """
        Delete a message.
        
        Args:
            record_id: Message ID to delete
            
        Returns:
            bool: True if deletion successful
        """
        try:
            query = "DELETE FROM messages WHERE id = ?"
            rows_affected = self.db.execute_update(query, (record_id,))
            
            if rows_affected > 0:
                self.db.logger.info(f"Message {record_id} deleted")
            
            return rows_affected > 0
            
        except Exception as e:
            self.db.logger.error(f"Failed to delete message {record_id}: {e}")
            return False
    
    def get_all(self, limit: Optional[int] = None, offset: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get all messages with parsed sources.
        
        Args:
            limit: Maximum number of messages to return
            offset: Number of messages to skip
            
        Returns:
            List[Dict[str, Any]]: List of message data
        """
        try:
            query = """
            SELECT id, session_id, message_type, content, sources, created_at
            FROM messages 
            ORDER BY created_at DESC
            """
            
            if limit:
                query += f" LIMIT {limit}"
                if offset:
                    query += f" OFFSET {offset}"
            
            results = self.db.execute_query(query)
            
            # Parse sources for all messages
            for message in results:
                if message['sources']:
                    try:
                        message['sources'] = json.loads(message['sources'])
                    except json.JSONDecodeError:
                        message['sources'] = None
            
            return results
            
        except Exception as e:
            self.db.logger.error(f"Failed to get all messages: {e}")
            return []
    
    def add_message(self, session_id: int, message_type: str, content: str, sources: Optional[List[str]] = None) -> int:
        """
        Add a message to a session (specialized method).
        
        Args:
            session_id: ID of the session
            message_type: Type of message ('human' or 'ai')
            content: Message content
            sources: Optional list of source files for AI responses
            
        Returns:
            int: ID of created message
            
        Raises:
            ValueError: If message_type is invalid or session doesn't exist
        """
        try:
            # Validate message type
            if message_type not in ['human', 'ai']:
                raise ValueError("message_type must be either 'human' or 'ai'")
            
            # Prepare message data
            message_data = {
                "session_id": session_id,
                "message_type": message_type,
                "content": content,
                "sources": sources
            }
            
            # Use the standard create method
            message_id = self.create(message_data)
            
            self.db.logger.info(f"Added {message_type} message to session {session_id} (ID: {message_id})")
            return message_id
            
        except Exception as e:
            self.db.logger.error(f"Failed to add message to session {session_id}: {e}")
            raise
    
    def add_human_message(self, session_id: int, content: str) -> int:
        """
        Add a human message to a session.
        
        Args:
            session_id: ID of the session
            content: Human message content
            
        Returns:
            int: ID of created message
        """
        return self.add_message(session_id, "human", content)
    
    def add_ai_message(self, session_id: int, content: str, sources: Optional[List[str]] = None) -> int:
        """
        Add an AI message to a session with optional sources.
        
        Args:
            session_id: ID of the session
            content: AI response content
            sources: Optional list of source files that informed the response
            
        Returns:
            int: ID of created message
        """
        return self.add_message(session_id, "ai", content, sources)
    
    def get_messages_by_session(self, session_id: int, message_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get messages for a specific session, optionally filtered by type.
        
        Args:
            session_id: Session ID
            message_type: Optional filter for message type ('human' or 'ai')
            
        Returns:
            List[Dict[str, Any]]: List of messages
        """
        try:
            query = """
            SELECT id, session_id, message_type, content, sources, created_at
            FROM messages 
            WHERE session_id = ?
            """
            params = [session_id]
            
            if message_type:
                query += " AND message_type = ?"
                params.append(message_type)
            
            query += " ORDER BY created_at ASC"
            
            results = self.db.execute_query(query, tuple(params))
            
            # Parse sources for all messages
            for message in results:
                if message['sources']:
                    try:
                        message['sources'] = json.loads(message['sources'])
                    except json.JSONDecodeError:
                        message['sources'] = None
            
            return results
            
        except Exception as e:
            self.db.logger.error(f"Failed to get messages for session {session_id}: {e}")
            return []
    
    def get_recent_messages(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get recent messages across all sessions.
        
        Args:
            limit: Maximum number of messages to return
            
        Returns:
            List[Dict[str, Any]]: List of recent messages
        """
        try:
            query = """
            SELECT 
                m.id,
                m.session_id,
                m.message_type,
                m.content,
                m.sources,
                m.created_at,
                s.session_name,
                u.username
            FROM messages m
            JOIN sessions s ON m.session_id = s.id
            JOIN users u ON s.user_id = u.id
            ORDER BY m.created_at DESC
            LIMIT ?
            """
            
            results = self.db.execute_query(query, (limit,))
            
            # Parse sources for all messages
            for message in results:
                if message['sources']:
                    try:
                        message['sources'] = json.loads(message['sources'])
                    except json.JSONDecodeError:
                        message['sources'] = None
            
            return results
            
        except Exception as e:
            self.db.logger.error(f"Failed to get recent messages: {e}")
            return []
    
    def search_messages(self, search_term: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Search messages by content.
        
        Args:
            search_term: Term to search for in message content
            limit: Maximum number of results to return
            
        Returns:
            List[Dict[str, Any]]: List of matching messages
        """
        try:
            query = """
            SELECT 
                m.id,
                m.session_id,
                m.message_type,
                m.content,
                m.sources,
                m.created_at,
                s.session_name,
                u.username
            FROM messages m
            JOIN sessions s ON m.session_id = s.id
            JOIN users u ON s.user_id = u.id
            WHERE m.content LIKE ?
            ORDER BY m.created_at DESC
            LIMIT ?
            """
            
            search_pattern = f"%{search_term}%"
            results = self.db.execute_query(query, (search_pattern, limit))
            
            # Parse sources for all messages
            for message in results:
                if message['sources']:
                    try:
                        message['sources'] = json.loads(message['sources'])
                    except json.JSONDecodeError:
                        message['sources'] = None
            
            return results
            
        except Exception as e:
            self.db.logger.error(f"Failed to search messages: {e}")
            return []
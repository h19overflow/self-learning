"""
User CRUD operations for chat history system.

This module provides comprehensive user management operations including
user authentication and session retrieval.
"""

import hashlib
from typing import Optional, Dict, Any, List
from .base_crud import BaseCRUD
from backend.chat_history.models.user import User, UserCreate


class UserCRUD(BaseCRUD):
    """
    CRUD operations for users with authentication support.
    """
    
    def _get_table_name(self) -> str:
        """Get table name for users."""
        return "users"
    
    def _hash_password(self, password: str) -> str:
        """
        Hash password using SHA-256.
        
        Args:
            password: Plain text password
            
        Returns:
            str: Hashed password
        """
        return hashlib.sha256(password.encode()).hexdigest()
    
    def create(self, data: Dict[str, Any]) -> int:
        """
        Create a new user with hashed password.
        
        Args:
            data: Dictionary containing user data (username, password)
            
        Returns:
            int: ID of created user
            
        Raises:
            ValueError: If username already exists
            Exception: If creation fails
        """
        try:
            # Validate input
            user_create = UserCreate(**data)
            
            # Check if username already exists
            if self.get_by_username(user_create.username):
                raise ValueError(f"Username '{user_create.username}' already exists")
            
            # Hash password and prepare data
            create_data = {
                "username": user_create.username,
                "password": self._hash_password(user_create.password)
            }
            
            query, params = self._build_insert_query(create_data)
            user_id = self.db.execute_insert(query, params)
            
            self.db.logger.info(f"User created: {user_create.username} (ID: {user_id})")
            return user_id
            
        except Exception as e:
            self.db.logger.error(f"Failed to create user: {e}")
            raise
    
    def get_by_id(self, record_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve a user by ID.
        
        Args:
            record_id: User ID
            
        Returns:
            Optional[Dict[str, Any]]: User data without password
        """
        try:
            query = "SELECT id, username, created_at FROM users WHERE id = ?"
            results = self.db.execute_query(query, (record_id,))
            return results[0] if results else None
            
        except Exception as e:
            self.db.logger.error(f"Failed to get user by ID {record_id}: {e}")
            return None
    
    def get_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a user by username.
        
        Args:
            username: Username to search for
            
        Returns:
            Optional[Dict[str, Any]]: User data without password
        """
        try:
            query = "SELECT id, username, created_at FROM users WHERE username = ?"
            results = self.db.execute_query(query, (username,))
            return results[0] if results else None
            
        except Exception as e:
            self.db.logger.error(f"Failed to get user by username {username}: {e}")
            return None
    
    def authenticate(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """
        Authenticate user with username and password.
        
        Args:
            username: Username
            password: Plain text password
            
        Returns:
            Optional[Dict[str, Any]]: User data if authentication successful
        """
        try:
            # Get user with password hash
            query = "SELECT id, username, password, created_at FROM users WHERE username = ?"
            results = self.db.execute_query(query, (username,))
            
            if not results:
                return None
                
            user_data = results[0]
            stored_hash = user_data['password']
            provided_hash = self._hash_password(password)
            
            if stored_hash == provided_hash:
                # Return user data without password
                return {
                    'id': user_data['id'],
                    'username': user_data['username'],
                    'created_at': user_data['created_at']
                }
            
            return None
            
        except Exception as e:
            self.db.logger.error(f"Authentication failed for {username}: {e}")
            return None
    
    def update(self, record_id: int, data: Dict[str, Any]) -> bool:
        """
        Update user information.
        
        Args:
            record_id: User ID
            data: Dictionary containing updated data
            
        Returns:
            bool: True if update successful
        """
        try:
            # Hash password if provided
            if 'password' in data:
                data['password'] = self._hash_password(data['password'])
            
            # Check username uniqueness if updating username
            if 'username' in data:
                existing = self.get_by_username(data['username'])
                if existing and existing['id'] != record_id:
                    raise ValueError(f"Username '{data['username']}' already exists")
            
            query, params = self._build_update_query(record_id, data)
            rows_affected = self.db.execute_update(query, params)
            
            return rows_affected > 0
            
        except Exception as e:
            self.db.logger.error(f"Failed to update user {record_id}: {e}")
            return False
    
    def delete(self, record_id: int) -> bool:
        """
        Delete a user (and cascade delete sessions/messages).
        
        Args:
            record_id: User ID to delete
            
        Returns:
            bool: True if deletion successful
        """
        try:
            query = "DELETE FROM users WHERE id = ?"
            rows_affected = self.db.execute_update(query, (record_id,))
            
            if rows_affected > 0:
                self.db.logger.info(f"User {record_id} deleted")
            
            return rows_affected > 0
            
        except Exception as e:
            self.db.logger.error(f"Failed to delete user {record_id}: {e}")
            return False
    
    def get_all(self, limit: Optional[int] = None, offset: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get all users without passwords.
        
        Args:
            limit: Maximum number of users to return
            offset: Number of users to skip
            
        Returns:
            List[Dict[str, Any]]: List of user data
        """
        try:
            query = "SELECT id, username, created_at FROM users ORDER BY created_at DESC"
            
            if limit:
                query += f" LIMIT {limit}"
                if offset:
                    query += f" OFFSET {offset}"
            
            return self.db.execute_query(query)
            
        except Exception as e:
            self.db.logger.error(f"Failed to get all users: {e}")
            return []
    
    def get_user_sessions(self, user_id: int) -> List[Dict[str, Any]]:
        """
        Get all sessions for a specific user.
        
        Args:
            user_id: ID of the user
            
        Returns:
            List[Dict[str, Any]]: List of user's sessions with message counts
        """
        try:
            # Query to get sessions with message counts
            query = """
            SELECT 
                s.id,
                s.user_id,
                s.session_name,
                s.created_at,
                s.updated_at,
                COUNT(m.id) as message_count
            FROM sessions s
            LEFT JOIN messages m ON s.id = m.session_id
            WHERE s.user_id = ?
            GROUP BY s.id, s.user_id, s.session_name, s.created_at, s.updated_at
            ORDER BY s.updated_at DESC
            """
            
            results = self.db.execute_query(query, (user_id,))
            
            self.db.logger.info(f"Retrieved {len(results)} sessions for user {user_id}")
            return results
            
        except Exception as e:
            self.db.logger.error(f"Failed to get sessions for user {user_id}: {e}")
            return []
    
    def get_user_stats(self, user_id: int) -> Dict[str, Any]:
        """
        Get statistics for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Dict[str, Any]: User statistics
        """
        try:
            # Get user info
            user = self.get_by_id(user_id)
            if not user:
                return {"error": "User not found"}
            
            # Get session count
            session_count = self.count("user_id = ?", (user_id,))
            
            # Get message count across all sessions
            message_count_query = """
            SELECT COUNT(m.id) as total_messages
            FROM messages m
            JOIN sessions s ON m.session_id = s.id
            WHERE s.user_id = ?
            """
            message_result = self.db.execute_query(message_count_query, (user_id,))
            message_count = message_result[0]['total_messages'] if message_result else 0
            
            return {
                "user_id": user_id,
                "username": user['username'],
                "created_at": user['created_at'],
                "total_sessions": session_count,
                "total_messages": message_count
            }
            
        except Exception as e:
            self.db.logger.error(f"Failed to get stats for user {user_id}: {e}")
            return {"error": str(e)}
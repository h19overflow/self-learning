"""
Session CRUD operations for chat history system.

This module provides comprehensive session management operations including
message retrieval and session analytics.
"""

from typing import Optional, Dict, Any, List
from .base_crud import BaseCRUD
from backend.chat_history.models.session import Session, SessionCreate


class SessionCRUD(BaseCRUD):
    """
    CRUD operations for sessions with message management support.
    """
    
    def _get_table_name(self) -> str:
        """Get table name for sessions."""
        return "sessions"
    
    def create(self, data: Dict[str, Any]) -> int:
        """
        Create a new session.
        
        Args:
            data: Dictionary containing session data (user_id, session_name)
            
        Returns:
            int: ID of created session
            
        Raises:
            ValueError: If user_id is invalid
            Exception: If creation fails
        """
        try:
            # Validate input
            session_create = SessionCreate(**data)
            
            # Verify user exists
            user_exists_query = "SELECT 1 FROM users WHERE id = ? LIMIT 1"
            user_result = self.db.execute_query(user_exists_query, (session_create.user_id,))
            if not user_result:
                raise ValueError(f"User with ID {session_create.user_id} does not exist")
            
            # Prepare session data
            create_data = {
                "user_id": session_create.user_id,
                "session_name": session_create.session_name
            }
            
            query, params = self._build_insert_query(create_data)
            session_id = self.db.execute_insert(query, params)
            
            self.db.logger.info(f"Session created: {session_create.session_name} (ID: {session_id})")
            return session_id
            
        except Exception as e:
            self.db.logger.error(f"Failed to create session: {e}")
            raise
    
    def get_by_id(self, record_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve a session by ID with message count.
        
        Args:
            record_id: Session ID
            
        Returns:
            Optional[Dict[str, Any]]: Session data with message count
        """
        try:
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
            WHERE s.id = ?
            GROUP BY s.id, s.user_id, s.session_name, s.created_at, s.updated_at
            """
            
            results = self.db.execute_query(query, (record_id,))
            return results[0] if results else None
            
        except Exception as e:
            self.db.logger.error(f"Failed to get session by ID {record_id}: {e}")
            return None
    
    def update(self, record_id: int, data: Dict[str, Any]) -> bool:
        """
        Update session information.
        
        Args:
            record_id: Session ID
            data: Dictionary containing updated data
            
        Returns:
            bool: True if update successful
        """
        try:
            # Only allow updating session_name
            allowed_fields = {'session_name'}
            update_data = {k: v for k, v in data.items() if k in allowed_fields}
            
            if not update_data:
                self.db.logger.warning(f"No valid fields to update for session {record_id}")
                return False
            
            query, params = self._build_update_query(record_id, update_data)
            rows_affected = self.db.execute_update(query, params)
            
            return rows_affected > 0
            
        except Exception as e:
            self.db.logger.error(f"Failed to update session {record_id}: {e}")
            return False
    
    def delete(self, record_id: int) -> bool:
        """
        Delete a session (and cascade delete messages).
        
        Args:
            record_id: Session ID to delete
            
        Returns:
            bool: True if deletion successful
        """
        try:
            query = "DELETE FROM sessions WHERE id = ?"
            rows_affected = self.db.execute_update(query, (record_id,))
            
            if rows_affected > 0:
                self.db.logger.info(f"Session {record_id} deleted")
            
            return rows_affected > 0
            
        except Exception as e:
            self.db.logger.error(f"Failed to delete session {record_id}: {e}")
            return False
    
    def get_all(self, limit: Optional[int] = None, offset: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get all sessions with message counts.
        
        Args:
            limit: Maximum number of sessions to return
            offset: Number of sessions to skip
            
        Returns:
            List[Dict[str, Any]]: List of session data
        """
        try:
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
            GROUP BY s.id, s.user_id, s.session_name, s.created_at, s.updated_at
            ORDER BY s.updated_at DESC
            """
            
            if limit:
                query += f" LIMIT {limit}"
                if offset:
                    query += f" OFFSET {offset}"
            
            return self.db.execute_query(query)
            
        except Exception as e:
            self.db.logger.error(f"Failed to get all sessions: {e}")
            return []
    
    def get_session_messages(self, session_id: int) -> List[Dict[str, Any]]:
        """
        Get all messages for a specific session in chronological order.
        
        Args:
            session_id: ID of the session
            
        Returns:
            List[Dict[str, Any]]: List of messages in the session
        """
        try:
            # Verify session exists
            if not self.exists(session_id):
                self.db.logger.warning(f"Session {session_id} does not exist")
                return []
            
            query = """
            SELECT 
                id,
                session_id,
                message_type,
                content,
                sources,
                created_at
            FROM messages 
            WHERE session_id = ?
            ORDER BY created_at ASC
            """
            
            results = self.db.execute_query(query, (session_id,))
            
            self.db.logger.info(f"Retrieved {len(results)} messages for session {session_id}")
            return results
            
        except Exception as e:
            self.db.logger.error(f"Failed to get messages for session {session_id}: {e}")
            return []
    
    def get_sessions_by_user(self, user_id: int, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get all sessions for a specific user.
        
        Args:
            user_id: User ID
            limit: Maximum number of sessions to return
            
        Returns:
            List[Dict[str, Any]]: List of user's sessions
        """
        try:
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
            
            if limit:
                query += f" LIMIT {limit}"
            
            return self.db.execute_query(query, (user_id,))
            
        except Exception as e:
            self.db.logger.error(f"Failed to get sessions for user {user_id}: {e}")
            return []
    
    def get_recent_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recently updated sessions across all users.
        
        Args:
            limit: Maximum number of sessions to return
            
        Returns:
            List[Dict[str, Any]]: List of recent sessions
        """
        try:
            query = """
            SELECT 
                s.id,
                s.user_id,
                u.username,
                s.session_name,
                s.created_at,
                s.updated_at,
                COUNT(m.id) as message_count
            FROM sessions s
            JOIN users u ON s.user_id = u.id
            LEFT JOIN messages m ON s.id = m.session_id
            GROUP BY s.id, s.user_id, u.username, s.session_name, s.created_at, s.updated_at
            ORDER BY s.updated_at DESC
            LIMIT ?
            """
            
            return self.db.execute_query(query, (limit,))
            
        except Exception as e:
            self.db.logger.error(f"Failed to get recent sessions: {e}")
            return []
    
    def get_session_stats(self, session_id: int) -> Dict[str, Any]:
        """
        Get statistics for a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Dict[str, Any]: Session statistics
        """
        try:
            # Get session info
            session = self.get_by_id(session_id)
            if not session:
                return {"error": "Session not found"}
            
            # Get message type counts
            type_count_query = """
            SELECT 
                message_type,
                COUNT(*) as count
            FROM messages
            WHERE session_id = ?
            GROUP BY message_type
            """
            type_counts = self.db.execute_query(type_count_query, (session_id,))
            
            # Convert to dict
            type_stats = {}
            total_messages = 0
            for row in type_counts:
                type_stats[row['message_type']] = row['count']
                total_messages += row['count']
            
            return {
                "session_id": session_id,
                "session_name": session['session_name'],
                "user_id": session['user_id'],
                "created_at": session['created_at'],
                "updated_at": session['updated_at'],
                "total_messages": total_messages,
                "human_messages": type_stats.get('human', 0),
                "ai_messages": type_stats.get('ai', 0)
            }
            
        except Exception as e:
            self.db.logger.error(f"Failed to get stats for session {session_id}: {e}")
            return {"error": str(e)}
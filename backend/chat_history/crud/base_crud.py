"""
Abstract base CRUD class for chat history system.

This module provides the abstract interface that all CRUD classes must implement,
ensuring consistent behavior across all database operations.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from backend.chat_history.database.connection import DatabaseConnection


class BaseCRUD(ABC):
    """
    Abstract base class for all CRUD operations.
    
    This class defines the standard interface that all CRUD classes must implement,
    providing consistent database operations across the chat history system.
    """
    
    def __init__(self, db_connection: DatabaseConnection):
        """
        Initialize CRUD class with database connection.
        
        Args:
            db_connection: Database connection manager
        """
        self.db = db_connection
        self.table_name = self._get_table_name()
    
    @abstractmethod
    def _get_table_name(self) -> str:
        """
        Get the table name for this CRUD class.
        
        Returns:
            str: Table name
        """
        pass
    
    @abstractmethod
    def create(self, data: Dict[str, Any]) -> int:
        """
        Create a new record in the database.
        
        Args:
            data: Dictionary containing record data
            
        Returns:
            int: ID of created record
            
        Raises:
            Exception: If creation fails
        """
        pass
    
    @abstractmethod
    def get_by_id(self, record_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve a record by its ID.
        
        Args:
            record_id: ID of the record to retrieve
            
        Returns:
            Optional[Dict[str, Any]]: Record data if found, None otherwise
        """
        pass
    
    @abstractmethod
    def update(self, record_id: int, data: Dict[str, Any]) -> bool:
        """
        Update an existing record.
        
        Args:
            record_id: ID of the record to update
            data: Dictionary containing updated data
            
        Returns:
            bool: True if update successful, False otherwise
        """
        pass
    
    @abstractmethod
    def delete(self, record_id: int) -> bool:
        """
        Delete a record by its ID.
        
        Args:
            record_id: ID of the record to delete
            
        Returns:
            bool: True if deletion successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_all(self, limit: Optional[int] = None, offset: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve all records from the table.
        
        Args:
            limit: Maximum number of records to return
            offset: Number of records to skip
            
        Returns:
            List[Dict[str, Any]]: List of all records
        """
        pass
    
    def exists(self, record_id: int) -> bool:
        """
        Check if a record exists by its ID.
        
        Args:
            record_id: ID of the record to check
            
        Returns:
            bool: True if record exists, False otherwise
        """
        try:
            query = f"SELECT 1 FROM {self.table_name} WHERE id = ? LIMIT 1"
            result = self.db.execute_query(query, (record_id,))
            return len(result) > 0
        except Exception:
            return False
    
    def count(self, where_clause: Optional[str] = None, params: tuple = ()) -> int:
        """
        Count records in the table.
        
        Args:
            where_clause: Optional WHERE clause (without 'WHERE' keyword)
            params: Parameters for the WHERE clause
            
        Returns:
            int: Number of records
        """
        try:
            query = f"SELECT COUNT(*) as count FROM {self.table_name}"
            if where_clause:
                query += f" WHERE {where_clause}"
            
            result = self.db.execute_query(query, params)
            return result[0]['count'] if result else 0
            
        except Exception as e:
            self.db.logger.error(f"Count operation failed: {e}")
            return 0
    
    def _build_insert_query(self, data: Dict[str, Any]) -> tuple:
        """
        Build INSERT query from data dictionary.
        
        Args:
            data: Dictionary containing record data
            
        Returns:
            tuple: (query_string, parameters)
        """
        columns = list(data.keys())
        placeholders = ', '.join(['?' for _ in columns])
        columns_str = ', '.join(columns)
        
        query = f"INSERT INTO {self.table_name} ({columns_str}) VALUES ({placeholders})"
        params = tuple(data.values())
        
        return query, params
    
    def _build_update_query(self, record_id: int, data: Dict[str, Any]) -> tuple:
        """
        Build UPDATE query from data dictionary.
        
        Args:
            record_id: ID of record to update
            data: Dictionary containing updated data
            
        Returns:
            tuple: (query_string, parameters)
        """
        set_clauses = [f"{column} = ?" for column in data.keys()]
        set_clause = ', '.join(set_clauses)
        
        query = f"UPDATE {self.table_name} SET {set_clause} WHERE id = ?"
        params = tuple(list(data.values()) + [record_id])
        
        return query, params
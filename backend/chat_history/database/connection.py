"""
Database connection manager for chat history system.

This module provides SQLite database connection management with automatic
schema creation and transaction handling.
"""

import sqlite3
import logging
import threading
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, Dict, Any, List


class DatabaseConnection:
    """
    SQLite database connection manager with connection pooling and schema management.
    """
    
    def __init__(self, db_path: str = "chat_history.db"):
        """
        Initialize database connection manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.logger = self._setup_logger()
        self._local = threading.local()
        
        # Ensure database directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database schema
        self._initialize_database()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for database operations."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _get_connection(self) -> sqlite3.Connection:
        """
        Get thread-local database connection.
        
        Returns:
            sqlite3.Connection: Database connection for current thread
        """
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=30.0
            )
            # Enable foreign key constraints
            self._local.connection.execute("PRAGMA foreign_keys = ON")
            # Set row factory to return dict-like objects
            self._local.connection.row_factory = sqlite3.Row
            
        return self._local.connection
    
    def _initialize_database(self):
        """Initialize database with schema if it doesn't exist."""
        try:
            schema_path = Path(__file__).parent / "schema.sql"
            
            if not schema_path.exists():
                raise FileNotFoundError(f"Schema file not found: {schema_path}")
            
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema_sql = f.read()
            
            with self.get_connection() as conn:
                # Execute schema in chunks (split by semicolon)
                for statement in schema_sql.split(';'):
                    statement = statement.strip()
                    if statement:  # Skip empty statements
                        conn.execute(statement)
                
                conn.commit()
                
            self.logger.info(f"Database initialized: {self.db_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections with automatic cleanup.
        
        Yields:
            sqlite3.Connection: Database connection
        """
        conn = self._get_connection()
        try:
            yield conn
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Database operation failed: {e}")
            raise
        finally:
            # Connection remains open for thread reuse
            pass
    
    @contextmanager 
    def get_transaction(self):
        """
        Context manager for database transactions with automatic commit/rollback.
        
        Yields:
            sqlite3.Connection: Database connection within transaction
        """
        conn = self._get_connection()
        try:
            conn.execute("BEGIN TRANSACTION")
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Transaction failed: {e}")
            raise
    
    def execute_query(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """
        Execute a SELECT query and return results as list of dictionaries.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            List[Dict[str, Any]]: Query results
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            self.logger.error(f"Query execution failed: {e}")
            raise
    
    def execute_insert(self, query: str, params: tuple = ()) -> int:
        """
        Execute an INSERT query and return the last inserted row ID.
        
        Args:
            query: SQL INSERT statement
            params: Insert parameters
            
        Returns:
            int: Last inserted row ID
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.execute(query, params)
                conn.commit()
                return cursor.lastrowid
                
        except Exception as e:
            self.logger.error(f"Insert execution failed: {e}")
            raise
    
    def execute_update(self, query: str, params: tuple = ()) -> int:
        """
        Execute an UPDATE/DELETE query and return number of affected rows.
        
        Args:
            query: SQL UPDATE/DELETE statement
            params: Query parameters
            
        Returns:
            int: Number of affected rows
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.execute(query, params)
                conn.commit()
                return cursor.rowcount
                
        except Exception as e:
            self.logger.error(f"Update execution failed: {e}")
            raise
    
    def close_all_connections(self):
        """Close all thread-local connections."""
        if hasattr(self._local, 'connection'):
            self._local.connection.close()
            delattr(self._local, 'connection')
            self.logger.info("Database connections closed")
    
    def get_database_info(self) -> Dict[str, Any]:
        """
        Get database information and statistics.
        
        Returns:
            Dict[str, Any]: Database information
        """
        try:
            with self.get_connection() as conn:
                # Get table information
                tables = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
                
                info = {
                    "database_path": str(self.db_path),
                    "database_size": self.db_path.stat().st_size if self.db_path.exists() else 0,
                    "tables": [table["name"] for table in tables]
                }
                
                # Get row counts for each table
                for table in tables:
                    table_name = table["name"]
                    count = conn.execute(f"SELECT COUNT(*) as count FROM {table_name}").fetchone()
                    info[f"{table_name}_count"] = count["count"]
                
                return info
                
        except Exception as e:
            self.logger.error(f"Failed to get database info: {e}")
            return {"error": str(e)}
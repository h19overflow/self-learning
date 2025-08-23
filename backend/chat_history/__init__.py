"""
Chat History Module for Self-Learning RAG System

This module provides comprehensive chat history management with SQLite storage,
including user management, session tracking, and message persistence.
"""

from .chat_history_manager import ChatHistoryManager

__all__ = ["ChatHistoryManager"]
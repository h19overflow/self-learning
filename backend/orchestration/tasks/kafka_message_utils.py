"""
Kafka message utilities for file-level processing pipeline.

Provides standardized message structures and validation for Kafka-based
orchestration of the document processing pipeline.
"""

from typing import Dict, Any, Optional, List
import asyncio
import uuid
from dataclasses import dataclass
from enum import Enum


class ProcessingStage(Enum):
    """Available processing stages."""
    PDF = "pdf"
    VLM = "vlm"
    CHUNK = "chunk"
    INGEST = "ingest"


class MessageStatus(Enum):
    """Message processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCESS = "success"
    FAILED = "failed"
    RETRY = "retry"


@dataclass
class ProcessingMessage:
    """Standardized message structure for file-level processing."""
    task_id: str
    file_path: str
    stage: ProcessingStage
    status: MessageStatus
    timestamp: float
    retry_count: int = 0
    max_retries: int = 2
    previous_result: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


def create_file_processing_message(
    file_path: str,
    stage: ProcessingStage,
    task_id: Optional[str] = None,
    previous_result: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> ProcessingMessage:
    """
    Create a standardized Kafka message for file-level processing.
    
    Args:
        file_path: Path to the file being processed
        stage: Processing stage
        task_id: Unique task identifier (auto-generated if None)
        previous_result: Result from previous stage
        metadata: Additional metadata
        
    Returns:
        ProcessingMessage object
    """
    if task_id is None:
        task_id = str(uuid.uuid4())
    
    return ProcessingMessage(
        task_id=task_id,
        file_path=file_path,
        stage=stage,
        status=MessageStatus.PENDING,
        timestamp=asyncio.get_event_loop().time(),
        previous_result=previous_result,
        metadata=metadata or {}
    )


def message_to_dict(message: ProcessingMessage) -> Dict[str, Any]:
    """
    Convert ProcessingMessage to dictionary for Kafka serialization.
    
    Args:
        message: ProcessingMessage object
        
    Returns:
        Dict representation of the message
    """
    return {
        "task_id": message.task_id,
        "file_path": message.file_path,
        "stage": message.stage.value,
        "status": message.status.value,
        "timestamp": message.timestamp,
        "retry_count": message.retry_count,
        "max_retries": message.max_retries,
        "previous_result": message.previous_result,
        "metadata": message.metadata,
        "error": message.error
    }


def dict_to_message(data: Dict[str, Any]) -> ProcessingMessage:
    """
    Convert dictionary to ProcessingMessage object.
    
    Args:
        data: Dictionary representation of message
        
    Returns:
        ProcessingMessage object
    """
    return ProcessingMessage(
        task_id=data["task_id"],
        file_path=data["file_path"],
        stage=ProcessingStage(data["stage"]),
        status=MessageStatus(data["status"]),
        timestamp=data["timestamp"],
        retry_count=data.get("retry_count", 0),
        max_retries=data.get("max_retries", 2),
        previous_result=data.get("previous_result"),
        metadata=data.get("metadata", {}),
        error=data.get("error")
    )


def validate_stage_transition(current_stage: ProcessingStage, next_stage: ProcessingStage) -> bool:
    """
    Validate that stage transition is allowed.
    
    Args:
        current_stage: Current processing stage
        next_stage: Next processing stage
        
    Returns:
        bool: True if transition is valid
    """
    valid_transitions = {
        ProcessingStage.PDF: [ProcessingStage.VLM, ProcessingStage.CHUNK],
        ProcessingStage.VLM: [ProcessingStage.CHUNK],
        ProcessingStage.CHUNK: [ProcessingStage.INGEST],
        ProcessingStage.INGEST: []
    }
    
    return next_stage in valid_transitions.get(current_stage, [])


def create_next_stage_message(
    current_message: ProcessingMessage,
    next_stage: ProcessingStage,
    result: Dict[str, Any]
) -> Optional[ProcessingMessage]:
    """
    Create next stage message based on current message and results.
    
    Args:
        current_message: Current processing message
        next_stage: Next processing stage
        result: Result from current stage
        
    Returns:
        ProcessingMessage for next stage or None if transition invalid
    """
    if not validate_stage_transition(current_message.stage, next_stage):
        return None
    
    return ProcessingMessage(
        task_id=current_message.task_id,
        file_path=current_message.file_path,
        stage=next_stage,
        status=MessageStatus.PENDING,
        timestamp=asyncio.get_event_loop().time(),
        retry_count=0,
        max_retries=current_message.max_retries,
        previous_result=result,
        metadata=current_message.metadata
    )


def should_retry_message(message: ProcessingMessage) -> bool:
    """
    Determine if a failed message should be retried.
    
    Args:
        message: ProcessingMessage that failed
        
    Returns:
        bool: True if should retry
    """
    return (
        message.status == MessageStatus.FAILED and
        message.retry_count < message.max_retries
    )


def create_retry_message(message: ProcessingMessage, error: str) -> ProcessingMessage:
    """
    Create a retry message from a failed message.
    
    Args:
        message: Failed ProcessingMessage
        error: Error message from failure
        
    Returns:
        New ProcessingMessage for retry
    """
    return ProcessingMessage(
        task_id=message.task_id,
        file_path=message.file_path,
        stage=message.stage,
        status=MessageStatus.RETRY,
        timestamp=asyncio.get_event_loop().time(),
        retry_count=message.retry_count + 1,
        max_retries=message.max_retries,
        previous_result=message.previous_result,
        metadata=message.metadata,
        error=error
    )


# HELPER FUNCTIONS

def get_kafka_topic_for_stage(stage: ProcessingStage) -> str:
    """
    Get Kafka topic name for a processing stage.
    
    Args:
        stage: Processing stage
        
    Returns:
        str: Kafka topic name
    """
    topic_mapping = {
        ProcessingStage.PDF: "file-processing-pdf",
        ProcessingStage.VLM: "file-processing-vlm",
        ProcessingStage.CHUNK: "file-processing-chunk",
        ProcessingStage.INGEST: "file-processing-ingest"
    }
    return topic_mapping[stage]


def create_batch_messages(
    file_paths: List[str],
    stage: ProcessingStage,
    metadata: Optional[Dict[str, Any]] = None
) -> List[ProcessingMessage]:
    """
    Create multiple processing messages for batch processing.
    
    Args:
        file_paths: List of file paths to process
        stage: Processing stage
        metadata: Common metadata for all messages
        
    Returns:
        List of ProcessingMessage objects
    """
    messages = []
    for file_path in file_paths:
        message = create_file_processing_message(
            file_path=file_path,
            stage=stage,
            metadata=metadata
        )
        messages.append(message)
    return messages
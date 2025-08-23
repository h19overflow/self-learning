-- Chat History Database Schema
-- SQLite database schema for chat history management system

-- Enable foreign key constraints
PRAGMA foreign_keys = ON;

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL CHECK(length(username) >= 3 AND length(username) <= 50),
    password TEXT NOT NULL CHECK(length(password) >= 6),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Sessions table  
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    session_name TEXT CHECK(length(session_name) <= 100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Messages table
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    message_type TEXT NOT NULL CHECK (message_type IN ('human', 'ai')),
    content TEXT NOT NULL CHECK(length(content) > 0),
    sources TEXT, -- JSON string for source files array
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

-- Indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_updated_at ON sessions(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_messages_type ON messages(message_type);

-- Trigger to update sessions.updated_at when messages are added
CREATE TRIGGER IF NOT EXISTS update_session_timestamp 
    AFTER INSERT ON messages
    FOR EACH ROW
BEGIN
    UPDATE sessions 
    SET updated_at = CURRENT_TIMESTAMP 
    WHERE id = NEW.session_id;
END;

-- Trigger to update sessions.updated_at when messages are updated
CREATE TRIGGER IF NOT EXISTS update_session_timestamp_on_message_update
    AFTER UPDATE ON messages
    FOR EACH ROW
BEGIN
    UPDATE sessions 
    SET updated_at = CURRENT_TIMESTAMP 
    WHERE id = NEW.session_id;
END;
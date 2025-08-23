"""
Example usage of the Chat History System.

This script demonstrates how to use the ChatHistoryManager for managing
users, sessions, and messages.
"""

from chat_history_manager import ChatHistoryManager


def main():
    """Demonstrate chat history system usage."""
    
    # Initialize the chat history manager
    chat_manager = ChatHistoryManager("example_chat_history.db")
    
    print("🚀 Chat History System Example")
    print("=" * 50)
    
    try:
        # 1. Create users
        print("\n📝 Creating Users...")
        user1_id = chat_manager.create_user("john_doe", "password123")
        user2_id = chat_manager.create_user("jane_smith", "securepass456")
        print(f"Created users: {user1_id}, {user2_id}")
        
        # 2. Authenticate user
        print("\n🔐 Authenticating User...")
        auth_result = chat_manager.authenticate_user("john_doe", "password123")
        print(f"Authentication successful: {auth_result['username']}")
        
        # 3. Create sessions
        print("\n📋 Creating Sessions...")
        session1_id = chat_manager.create_session(user1_id, "Machine Learning Questions")
        session2_id = chat_manager.create_session(user1_id, "Python Programming")
        session3_id = chat_manager.create_session(user2_id, "Data Science Discussion")
        print(f"Created sessions: {session1_id}, {session2_id}, {session3_id}")
        
        # 4. Add conversation pairs
        print("\n💬 Adding Conversations...")
        
        # Session 1: ML Questions
        chat_manager.add_conversation_pair(
            session1_id,
            "What is machine learning?",
            "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every task.",
            ["ml_textbook.pdf", "ai_introduction.pdf"]
        )
        
        chat_manager.add_conversation_pair(
            session1_id,
            "How does neural attention work?",
            "Neural attention is a mechanism that allows models to focus on specific parts of the input when making predictions. It computes attention weights that determine the importance of different input elements.",
            ["attention_paper.pdf", "transformer_guide.pdf"]
        )
        
        # Session 2: Python Programming
        chat_manager.add_human_message(session2_id, "How do I handle exceptions in Python?")
        chat_manager.add_ai_message(
            session2_id, 
            "Python uses try-except blocks to handle exceptions. You can catch specific exceptions and handle them gracefully to prevent your program from crashing.",
            ["python_docs.pdf"]
        )
        
        # Session 3: Data Science
        chat_manager.add_conversation_pair(
            session3_id,
            "What's the difference between supervised and unsupervised learning?",
            "Supervised learning uses labeled data to train models, while unsupervised learning finds patterns in data without labels. Supervised learning includes classification and regression, while unsupervised includes clustering and dimensionality reduction."
        )
        
        # 5. Retrieve user sessions
        print("\n📊 User Sessions:")
        user1_sessions = chat_manager.get_user_sessions(user1_id)
        for session in user1_sessions:
            print(f"  - {session['session_name']}: {session['message_count']} messages")
        
        # 6. Get session conversation
        print("\n🗨️  ML Session Conversation:")
        ml_conversation = chat_manager.get_session_conversation(session1_id)
        for message in ml_conversation:
            speaker = "Human" if message['message_type'] == 'human' else "AI"
            print(f"  {speaker}: {message['content'][:60]}...")
            if message['sources']:
                print(f"    Sources: {', '.join(message['sources'])}")
        
        # 7. Search messages
        print("\n🔍 Search Results for 'learning':")
        search_results = chat_manager.search_messages("learning", limit=5)
        for result in search_results:
            print(f"  - [{result['username']}] {result['content'][:50]}...")
        
        # 8. Get statistics
        print("\n📈 System Statistics:")
        stats = chat_manager.get_system_stats()
        print(f"  - Total Users: {stats['total_users']}")
        print(f"  - Total Sessions: {stats['total_sessions']}")
        print(f"  - Total Messages: {stats['total_messages']}")
        print(f"  - Human Messages: {stats['human_messages']}")
        print(f"  - AI Messages: {stats['ai_messages']}")
        
        # 9. User statistics
        print(f"\n👤 User Statistics for {auth_result['username']}:")
        user_stats = chat_manager.get_user_stats(user1_id)
        print(f"  - Sessions: {user_stats['total_sessions']}")
        print(f"  - Messages: {user_stats['total_messages']}")
        
        print("\n✅ Chat History System Demo Complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        
    finally:
        # Clean up
        chat_manager.close_database()


if __name__ == "__main__":
    main()
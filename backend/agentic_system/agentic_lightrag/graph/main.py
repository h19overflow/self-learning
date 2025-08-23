"""
Agentic LightRAG Workflow Demonstration
"""

import asyncio
from .state import AgenticLightRAGState
from .nodes.query_analysis_node import run_query_analysis_node
from .nodes.retrieval_node import run_retrieval_node
from .nodes.answering_node import run_answering_node
from .nodes.corrective_node import run_corrective_node, needs_correction
import weave

weave.init('LIGHT_RAG_AGENTIC_SYSTEM')
class AgenticLightRAGWorkflow:
    """
    Main workflow orchestrator that manages the complete agentic pipeline.
    """
    
    def __init__(self):
        """Initialize the workflow."""
        pass
    
    async def process_question(self, question: str) -> AgenticLightRAGState:
        """
        Process a user question through the complete agentic workflow.
        
        Args:
            question: User's question to process
            
        Returns:
            AgenticLightRAGState: Final state with answer
        """
        print("=" * 80)
        print(f"🤖 AGENTIC LIGHTRAG WORKFLOW")
        print("=" * 80)
        print(f"📝 Question: {question}")
        print("-" * 80)
       
        # Initialize state
        state = AgenticLightRAGState(question=question, messages=[])
        print(f"🔍 DEBUG: Initial state created - type: {type(state)}")
        print(f"🔍 DEBUG: State question: {state.question}")
        
        try:
            # Step 1: Query Analysis
            print("\n🔍 STEP 1: Query Analysis")
            print(f"🔍 DEBUG: About to run query analysis node...")
            state = await run_query_analysis_node(state)
            print(f"🔍 DEBUG: Query analysis complete - state type: {type(state)}")
            
            if state.query_analysis:
                strategy = state.query_analysis.strategy
                print(f"   ✅ Chunk-Top-K: {strategy.chunk_top_k}")
                print(f"   ✅ Analysis: {strategy.analysis[:100]}...")
            else:
                print(f"🔍 DEBUG: No query analysis found in state")
            
            # Step 2: Context Retrieval
            print("\n🔎 STEP 2: Context Retrieval")
            print(f"🔍 DEBUG: About to run retrieval node...")
            state = await run_retrieval_node(state)
            print(f"🔍 DEBUG: Retrieval complete - state type: {type(state)}")
            
            if state.context:
                print(f"   ✅ Retrieved: {len(state.context)} characters")
                print(f"   ✅ Preview: {state.context[:200]}...")
            else:
                print(f"🔍 DEBUG: No context found in state")
            
            # Step 3: Answer Generation with Corrective Loop
            max_retries = 2
            retry_attempt = 0
            
            while retry_attempt <= max_retries:
                # Generate answer
                if retry_attempt == 0:
                    print("\n🎓 STEP 3: Educational Answer Generation")
                else:
                    print(f"\n🔄 STEP 3.{retry_attempt}: Corrective Answer Generation (Retry {retry_attempt})")
                
                print(f"🔍 DEBUG: About to run answering node...")
                print(f"🔍 DEBUG: State before answering - question: {state.question}")
                print(f"🔍 DEBUG: State before answering - context length: {len(state.context) if state.context else 0}")
                
                state = await run_answering_node(state)
                print(f"🔍 DEBUG: Answering complete - state type: {type(state)}")
                
                if state.answer:
                    print(f"   ✅ Answer Length: {len(state.answer)} characters")
                    print(f"   ✅ Answer Preview: {state.answer[:200]}...")
                    
                    # Check if correction is needed
                    if needs_correction(state.answer) and retry_attempt < max_retries:
                        print(f"\n⚠️  INSUFFICIENT CONTEXT DETECTED")
                        print(f"   🔄 Triggering corrective retrieval (attempt {retry_attempt + 1}/{max_retries})")
                        
                        # Run corrective node to enhance retrieval
                        state = await run_corrective_node(state, retry_attempt + 1)
                        retry_attempt += 1
                        continue
                    elif needs_correction(state.answer) and retry_attempt >= max_retries:
                        print(f"\n⚠️  INSUFFICIENT CONTEXT DETECTED - Maximum retries reached")
                        print(f"   ℹ️  Proceeding with best available answer")
                        break
                    else:
                        print(f"   ✅ Answer appears complete - no correction needed")
                        break
                else:
                    print(f"🔍 DEBUG: No answer found in state")
                    if retry_attempt < max_retries:
                        print(f"   🔄 Retrying with enhanced retrieval...")
                        state = await run_corrective_node(state, retry_attempt + 1)
                        retry_attempt += 1
                        continue
                    else:
                        break
            
            print(f"\n📊 RETRIEVAL SUMMARY:")
            if retry_attempt > 0:
                print(f"   🔄 Total retrieval attempts: {retry_attempt + 1}")
                print(f"   📈 Final context size: {len(state.context) if state.context else 0} characters")
            else:
                print(f"   ✅ Single retrieval successful")
            
            print("\n" + "=" * 80)
            print("✨ WORKFLOW COMPLETE")
            print("=" * 80)
            
            return state
            
        except Exception as e:
            print(f"\n❌ WORKFLOW ERROR: {type(e).__name__}: {e}")
            import traceback
            print(f"🔍 DEBUG: Full error traceback:\n{traceback.format_exc()}")
            
            state.answer = f"I apologize, but I encountered an error while processing your question: '{e}'. Please try rephrasing your question or provide more context."
            return state
# HELPER FUNCTIONS

async def run_single_question(question: str) -> AgenticLightRAGState:
    """
    Run the workflow for a single question.
    
    Args:
        question: User question
        
    Returns:
        AgenticLightRAGState: Final state
    """
    workflow = AgenticLightRAGWorkflow()
    return await workflow.process_question(question)

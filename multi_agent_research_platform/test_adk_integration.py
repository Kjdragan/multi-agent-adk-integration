#!/usr/bin/env python3
"""
Test script to verify ADK integration with real task execution
"""
import asyncio
import sys
import os
import time
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_adk_integration():
    """Test ADK integration with real task execution"""
    try:
        # Load environment
        load_dotenv()
        
        # Import required modules
        from agents.factory import AgentFactory
        from agents.llm_agent import LLMRole
        from services import create_development_services
        
        print("=== ADK Integration Test ===")
        print(f"GOOGLE_GENAI_USE_VERTEXAI: {os.getenv('GOOGLE_GENAI_USE_VERTEXAI')}")
        print(f"GOOGLE_CLOUD_PROJECT: {os.getenv('GOOGLE_CLOUD_PROJECT')}")
        print(f"GOOGLE_API_KEY: {'SET' if os.getenv('GOOGLE_API_KEY') else 'NOT SET'}")
        
        # Create services
        print("\n1. Creating services...")
        services = create_development_services()
        print("✓ Services created successfully")
        
        # Create agent factory
        print("\n2. Creating agent factory...")
        factory = AgentFactory()
        print("✓ Agent factory created successfully")
        
        # Create LLM agent
        print("\n3. Creating LLM agent...")
        agent = factory.create_llm_agent(
            role=LLMRole.RESEARCHER,
            session_service=services.session_service,
            memory_service=services.memory_service,
            artifact_service=services.artifact_service
        )
        print(f"✓ Agent created: {agent.agent_id}")
        print(f"✓ Agent role: {agent.role}")
        
        # Execute a simple task
        print("\n4. Executing simple task...")
        test_task = "What is 2+2? Please provide just the answer."
        
        start_time = time.time()
        result = await agent.execute_task(test_task)
        execution_time = time.time() - start_time
        
        print(f"✓ Task completed in {execution_time:.2f} seconds")
        print(f"✓ Success: {result.success}")
        
        if result.success:
            print(f"✓ Result: {result.result}")
            print(f"✓ Agent ID: {result.agent_id}")
            print(f"✓ Tools used: {result.tools_used}")
            print(f"✓ Execution time: {result.execution_time_ms}ms")
            
            # Check if it's a real ADK response (not simulation)
            if "adk_runner" in result.tools_used:
                print("✓ REAL ADK INTEGRATION CONFIRMED!")
                return True
            else:
                print("✗ Warning: ADK runner not used in tools")
                return False
        else:
            print(f"✗ Task failed: {result.error}")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_adk_integration())
    print(f"\n=== TEST RESULT: {'PASSED' if success else 'FAILED'} ===")
    sys.exit(0 if success else 1)
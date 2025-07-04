# Basic integration tests for core agent functionality
# Tests that the key components work together properly

import pytest
from unittest.mock import Mock, AsyncMock

from src.agents.base import AgentResult, AgentCapability, AgentType, AgentRegistry
from src.agents.llm_agent import LLMRole, LLMAgentConfig  
from src.agents.factory import AgentFactory


class TestBasicAgentIntegration:
    """Test basic integration between agent components."""
    
    def test_agent_factory_creation(self):
        """Test that agent factory can be created."""
        factory = AgentFactory()
        assert factory is not None
        assert hasattr(factory, 'create_llm_agent')
        assert hasattr(factory, 'default_llm_configs')
    
    def test_llm_agent_config_creation(self):
        """Test LLM agent config creation with different roles."""
        roles_to_test = [
            LLMRole.RESEARCHER,
            LLMRole.ANALYST, 
            LLMRole.SYNTHESIZER,
            LLMRole.CRITIC
        ]
        
        for role in roles_to_test:
            config = LLMAgentConfig(role=role)
            assert config.role == role
            assert config.model is None  # Auto-selection
    
    def test_agent_registry_integration(self):
        """Test agent registry operations."""
        # Clear registry
        AgentRegistry.clear()
        
        # Create mock agent
        mock_agent = Mock()
        mock_agent.agent_id = "test_integration_agent"
        mock_agent.agent_type = AgentType.LLM
        mock_agent.is_active = True
        mock_agent.get_capabilities.return_value = [
            AgentCapability.RESEARCH, 
            AgentCapability.ANALYSIS
        ]
        
        # Register and test
        AgentRegistry.register(mock_agent)
        
        # Verify registration
        retrieved = AgentRegistry.get_agent("test_integration_agent")
        assert retrieved == mock_agent
        
        # Test capability search
        research_agents = AgentRegistry.get_agents_by_capability(AgentCapability.RESEARCH)
        assert len(research_agents) == 1
        assert research_agents[0] == mock_agent
        
        # Test status
        status = AgentRegistry.get_registry_status()
        assert status["total_agents"] == 1
        assert status["active_agents"] == 1
        
        # Cleanup
        AgentRegistry.clear()
        assert AgentRegistry.get_registry_status()["total_agents"] == 0
    
    def test_agent_result_integration(self):
        """Test AgentResult creation and serialization."""
        result = AgentResult(
            agent_id="integration_test_agent",
            agent_type=AgentType.LLM,
            task="Integration test task",
            success=True,
            result="Integration test completed",
            execution_time_ms=150.5,
            metadata={"test_type": "integration"},
            tools_used=["mock_tool"],
            memory_accessed=True
        )
        
        # Test basic properties
        assert result.agent_id == "integration_test_agent"
        assert result.agent_type == AgentType.LLM
        assert result.success is True
        assert result.execution_time_ms == 150.5
        
        # Test serialization
        result_dict = result.to_dict()
        assert result_dict["agent_type"] == "llm"
        assert result_dict["success"] is True
        assert result_dict["metadata"]["test_type"] == "integration"
        assert result_dict["tools_used"] == ["mock_tool"]
        assert result_dict["memory_accessed"] is True
    
    def test_capability_system_integration(self):
        """Test the capability system works properly."""
        # Test all defined capabilities
        capabilities = [
            AgentCapability.REASONING,
            AgentCapability.RESEARCH,
            AgentCapability.ANALYSIS,
            AgentCapability.SYNTHESIS,
            AgentCapability.PLANNING,
            AgentCapability.EXECUTION,
            AgentCapability.COMMUNICATION,
            AgentCapability.LEARNING,
            AgentCapability.TOOL_USE,
            AgentCapability.MEMORY_ACCESS,
            AgentCapability.CONTEXT_MANAGEMENT,
            AgentCapability.FACT_CHECKING,
            AgentCapability.CONTENT_GENERATION,
            AgentCapability.DECISION_MAKING
        ]
        
        for capability in capabilities:
            assert isinstance(capability.value, str)
            assert len(capability.value) > 0
        
        # Test capability matching
        mock_agent = Mock()
        mock_agent.get_capabilities.return_value = [
            AgentCapability.RESEARCH,
            AgentCapability.ANALYSIS,
            AgentCapability.SYNTHESIS
        ]
        
        # Mock the can_handle_task method
        def mock_can_handle_task(task, required_capabilities):
            agent_caps = set(mock_agent.get_capabilities())
            required_caps = set(required_capabilities)
            return required_caps.issubset(agent_caps)
        
        mock_agent.can_handle_task = mock_can_handle_task
        
        # Test capability matching
        assert mock_agent.can_handle_task("", [AgentCapability.RESEARCH])
        assert mock_agent.can_handle_task("", [AgentCapability.RESEARCH, AgentCapability.ANALYSIS])
        assert not mock_agent.can_handle_task("", [AgentCapability.RESEARCH, AgentCapability.PLANNING])
    
    @pytest.mark.asyncio 
    async def test_mock_agent_workflow(self):
        """Test a simple mock workflow between components."""
        AgentRegistry.clear()
        
        # Create mock agents
        researcher = Mock()
        researcher.agent_id = "researcher_1"
        researcher.agent_type = AgentType.LLM
        researcher.is_active = True
        researcher.get_capabilities.return_value = [AgentCapability.RESEARCH]
        researcher.execute_task = AsyncMock(return_value=AgentResult(
            agent_id="researcher_1",
            agent_type=AgentType.LLM,
            task="Research task",
            success=True,
            result="Research completed",
            execution_time_ms=100
        ))
        
        analyst = Mock()
        analyst.agent_id = "analyst_1"
        analyst.agent_type = AgentType.LLM
        analyst.is_active = True
        analyst.get_capabilities.return_value = [AgentCapability.ANALYSIS]
        analyst.execute_task = AsyncMock(return_value=AgentResult(
            agent_id="analyst_1",
            agent_type=AgentType.LLM,
            task="Analysis task",
            success=True,
            result="Analysis completed",
            execution_time_ms=200
        ))
        
        # Register agents
        AgentRegistry.register(researcher)
        AgentRegistry.register(analyst)
        
        # Test workflow
        research_result = await researcher.execute_task("Do research")
        assert research_result.success
        assert research_result.result == "Research completed"
        
        analysis_result = await analyst.execute_task("Analyze research")
        assert analysis_result.success  
        assert analysis_result.result == "Analysis completed"
        
        # Verify registry state
        assert len(AgentRegistry.get_all_agents()) == 2
        research_agents = AgentRegistry.get_agents_by_capability(AgentCapability.RESEARCH)
        assert len(research_agents) == 1
        
        analysis_agents = AgentRegistry.get_agents_by_capability(AgentCapability.ANALYSIS)
        assert len(analysis_agents) == 1
        
        # Cleanup
        AgentRegistry.clear()


class TestErrorHandling:
    """Test error handling in integration scenarios."""
    
    def test_agent_result_with_errors(self):
        """Test AgentResult error handling."""
        error_result = AgentResult(
            agent_id="error_agent",
            agent_type=AgentType.LLM,
            task="Failing task",
            success=False,
            error="Test error occurred",
            execution_time_ms=50
        )
        
        assert error_result.success is False
        assert error_result.error == "Test error occurred"
        assert error_result.result is None
        
        # Test serialization of error
        error_dict = error_result.to_dict()
        assert error_dict["success"] is False
        assert error_dict["error"] == "Test error occurred"
    
    def test_registry_error_handling(self):
        """Test registry handles errors gracefully."""
        AgentRegistry.clear()
        
        # Test getting non-existent agent
        result = AgentRegistry.get_agent("non_existent_agent")
        assert result is None
        
        # Test unregistering non-existent agent
        success = AgentRegistry.unregister("non_existent_agent")
        assert success is False
        
        # Test empty queries
        agents = AgentRegistry.get_agents_by_type(AgentType.LLM)
        assert agents == []
        
        agents = AgentRegistry.get_agents_by_capability(AgentCapability.RESEARCH)
        assert agents == []


class TestPerformanceIntegration:
    """Test performance-related integration aspects."""
    
    def test_agent_result_timing(self):
        """Test that execution timing is properly tracked."""
        result = AgentResult(
            agent_id="timing_agent",
            agent_type=AgentType.LLM,
            task="Timing test",
            success=True,
            result="Done",
            execution_time_ms=1234.56
        )
        
        assert result.execution_time_ms == 1234.56
        
        # Test serialization preserves timing
        result_dict = result.to_dict()
        assert result_dict["execution_time_ms"] == 1234.56
    
    def test_registry_performance_tracking(self):
        """Test registry status for performance monitoring."""
        AgentRegistry.clear()
        
        # Add multiple agents
        for i in range(5):
            mock_agent = Mock()
            mock_agent.agent_id = f"perf_agent_{i}"
            mock_agent.agent_type = AgentType.LLM
            mock_agent.is_active = (i % 2 == 0)  # Alternate active/inactive
            mock_agent.get_capabilities.return_value = [AgentCapability.RESEARCH]
            
            AgentRegistry.register(mock_agent)
        
        status = AgentRegistry.get_registry_status()
        assert status["total_agents"] == 5
        assert status["active_agents"] == 3  # 0, 2, 4 are active
        
        # Test agent distribution
        assert status["agents_by_type"]["llm"] == 5
        assert status["agents_by_capability"]["research"] == 5
        
        AgentRegistry.clear()
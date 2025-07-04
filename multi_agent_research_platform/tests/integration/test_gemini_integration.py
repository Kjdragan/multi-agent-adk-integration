"""
Integration tests for Gemini 2.5 model selection and agent integration.

Tests the complete workflow from task analysis to agent creation and execution
with the new Gemini 2.5 model capabilities.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

from multi_agent_research_platform.src.config.gemini_models import (
    GeminiModel, TaskComplexity, ModelSelector, 
    analyze_task_complexity, select_model_for_task,
    StructuredOutputConfig, ThinkingBudgetConfig
)
from multi_agent_research_platform.src.agents.factory import AgentFactory
from multi_agent_research_platform.src.agents.llm_agent import LLMRole, LLMAgentConfig
from multi_agent_research_platform.src.agents.orchestrator import AgentOrchestrator, OrchestrationStrategy
from multi_agent_research_platform.src.platform_logging.logger import RunLogger


class TestGeminiModelIntegration:
    """Test Gemini 2.5 model integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.logger = Mock(spec=RunLogger)
        self.factory = AgentFactory(logger=self.logger)
        self.orchestrator = AgentOrchestrator(logger=self.logger)
        self.model_selector = ModelSelector()
    
    def test_task_complexity_analysis(self):
        """Test task complexity analysis accuracy."""
        test_cases = [
            {
                "task": "What is 2+2?",
                "expected": TaskComplexity.SIMPLE
            },
            {
                "task": "Explain the basic concept of photosynthesis",
                "expected": TaskComplexity.MEDIUM
            },
            {
                "task": "Conduct a comprehensive analysis of the economic implications of artificial intelligence on global labor markets, including sector-specific impacts, policy recommendations, and long-term societal effects",
                "expected": TaskComplexity.COMPLEX
            },
            {
                "task": "Perform a critical strategic risk assessment for a multinational corporation considering a merger in the semiconductor industry, analyzing regulatory compliance across multiple jurisdictions, competitive landscape, technological disruption risks, and financial implications",
                "expected": TaskComplexity.CRITICAL
            }
        ]
        
        for case in test_cases:
            detected = analyze_task_complexity(case["task"])
            assert detected == case["expected"], f"Task: {case['task'][:50]}... Expected: {case['expected']}, Got: {detected}"
    
    def test_model_selection_for_different_preferences(self):
        """Test model selection with different preferences."""
        task = "Analyze the impact of climate change on global supply chains"
        
        # Test speed priority
        speed_config = select_model_for_task(
            task=task,
            preferences={"priority_speed": True}
        )
        assert speed_config.model.speed_level >= 7, "Speed priority should select high-speed model"
        
        # Test cost priority
        cost_config = select_model_for_task(
            task=task,
            preferences={"priority_cost": True}
        )
        assert cost_config.model.cost_level <= 5, "Cost priority should select low-cost model"
        
        # Test thinking enabled
        thinking_config = select_model_for_task(
            task=task,
            preferences={"enable_thinking": True}
        )
        assert thinking_config.thinking_config is not None, "Should enable thinking config"
        assert thinking_config.thinking_config.enabled, "Thinking should be enabled"
        
        # Test structured output
        structured_config = select_model_for_task(
            task=task,
            preferences={"enable_structured_output": True}
        )
        assert structured_config.structured_output is not None, "Should enable structured output"
        assert structured_config.structured_output.enabled, "Structured output should be enabled"
    
    def test_agent_factory_gemini_integration(self):
        """Test agent factory creates agents with proper Gemini configuration."""
        # Test auto-optimization enabled
        agent = self.factory.create_llm_agent(
            role=LLMRole.RESEARCHER,
            auto_optimize_model=True,
            enable_thinking=True,
            enable_structured_output=True
        )
        
        assert agent.config.auto_optimize_model, "Auto-optimization should be enabled"
        assert agent.config.enable_thinking, "Thinking should be enabled"
        assert agent.config.enable_structured_output, "Structured output should be enabled"
        assert agent.config.role == LLMRole.RESEARCHER, "Role should be set correctly"
        
        # Test specific model selection
        agent_flash = self.factory.create_llm_agent(
            role=LLMRole.COMMUNICATOR,
            model=GeminiModel.FLASH,
            auto_optimize_model=False
        )
        
        assert agent_flash.config.model == GeminiModel.FLASH, "Should use specified model"
        assert not agent_flash.config.auto_optimize_model, "Auto-optimization should be disabled"
    
    def test_task_optimized_agent_creation(self):
        """Test task-optimized agent creation."""
        # Simple task should get fast model
        simple_agent = self.factory.create_task_optimized_agent(
            task_description="What is machine learning?",
            role=LLMRole.GENERALIST
        )
        
        assert simple_agent is not None, "Should create agent successfully"
        assert simple_agent.config.auto_optimize_model, "Should enable auto-optimization"
        
        # Complex task should get thinking enabled
        complex_agent = self.factory.create_task_optimized_agent(
            task_description="Conduct comprehensive research on quantum computing applications in cryptography with detailed analysis of current limitations and future potential",
            role=LLMRole.RESEARCHER
        )
        
        assert complex_agent is not None, "Should create agent successfully"
        assert complex_agent.config.enable_thinking, "Should enable thinking for complex tasks"
    
    def test_research_team_optimization(self):
        """Test research team creation with task optimization."""
        research_task = "Analyze the economic and environmental impact of renewable energy adoption in developing countries"
        
        # Test different team sizes
        for team_size in ["minimal", "standard", "comprehensive"]:
            team = self.factory.create_research_team_for_task(
                task_description=research_task,
                team_size=team_size
            )
            
            assert len(team) > 0, f"Should create non-empty {team_size} team"
            
            # Verify all agents have proper configuration
            for agent in team:
                assert hasattr(agent, 'config'), "Agent should have config"
                assert agent.config.auto_optimize_model, "Should enable auto-optimization"
                
            # Check expected team sizes
            if team_size == "minimal":
                assert len(team) == 2, "Minimal team should have 2 agents"
            elif team_size == "standard":
                assert len(team) == 3, "Standard team should have 3 agents"
            elif team_size == "comprehensive":
                assert len(team) == 5, "Comprehensive team should have 5 agents"
    
    def test_structured_output_schemas(self):
        """Test structured output schema generation."""
        # Test analysis schema
        analysis_schema = StructuredOutputConfig.get_analysis_schema()
        assert "type" in analysis_schema, "Schema should have type"
        assert "properties" in analysis_schema, "Schema should have properties"
        assert "summary" in analysis_schema["properties"], "Should include summary field"
        assert "key_findings" in analysis_schema["properties"], "Should include key_findings field"
        
        # Test research schema
        research_schema = StructuredOutputConfig.get_research_schema()
        assert "type" in research_schema, "Schema should have type"
        assert "properties" in research_schema, "Schema should have properties"
        assert "research_topic" in research_schema["properties"], "Should include research_topic field"
        assert "findings" in research_schema["properties"], "Should include findings field"
    
    def test_thinking_budget_configuration(self):
        """Test thinking budget configuration for different complexities."""
        thinking_config = ThinkingBudgetConfig()
        
        # Test budget calculation for different complexities
        simple_budget = thinking_config.get_budget_for_complexity(TaskComplexity.SIMPLE)
        medium_budget = thinking_config.get_budget_for_complexity(TaskComplexity.MEDIUM)
        complex_budget = thinking_config.get_budget_for_complexity(TaskComplexity.COMPLEX)
        critical_budget = thinking_config.get_budget_for_complexity(TaskComplexity.CRITICAL)
        
        # Should increase with complexity
        assert simple_budget <= medium_budget, "Medium budget should be >= simple budget"
        assert medium_budget <= complex_budget, "Complex budget should be >= medium budget"
        assert complex_budget <= critical_budget, "Critical budget should be >= complex budget"
    
    def test_cost_estimation(self):
        """Test cost estimation for different configurations."""
        # Simple task with speed priority
        simple_config = select_model_for_task(
            task="What is Python?",
            preferences={"priority_speed": True}
        )
        
        simple_costs = simple_config.estimate_cost_factors()
        assert "total_factor" in simple_costs, "Should include total cost factor"
        assert simple_costs["total_factor"] > 0, "Cost factor should be positive"
        
        # Complex task with thinking and structured output
        complex_config = select_model_for_task(
            task="Comprehensive analysis of AI impact on society",
            preferences={
                "enable_thinking": True,
                "enable_structured_output": True
            }
        )
        
        complex_costs = complex_config.estimate_cost_factors()
        
        # Complex task should generally cost more
        assert complex_costs["total_factor"] >= simple_costs["total_factor"], \
            "Complex task should have higher or equal cost factor"
    
    @pytest.mark.asyncio
    async def test_agent_orchestration_with_optimized_agents(self):
        """Test orchestration with task-optimized agents."""
        # Create optimized research team
        research_task = "Research the latest developments in renewable energy technology"
        team = self.factory.create_research_team_for_task(
            task_description=research_task,
            team_size="standard"
        )
        
        # Mock the agent execution to avoid API calls
        for agent in team:
            agent.execute_task = AsyncMock(return_value=Mock(
                success=True,
                result=f"Mock result from {agent.name}",
                execution_time_ms=100,
                metadata={"mock": True}
            ))
        
        # Mock the agent selection to return our test team
        self.orchestrator._select_agents = AsyncMock(return_value=team)
        
        # Test orchestration
        result = await self.orchestrator.orchestrate_task(
            task=research_task,
            strategy=OrchestrationStrategy.CONSENSUS
        )
        
        assert result is not None, "Should return orchestration result"
        assert len(result.agents_used) > 0, "Should use multiple agents"


class TestGeminiIntegrationErrorHandling:
    """Test error handling in Gemini integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.factory = AgentFactory()
    
    def test_invalid_task_complexity(self):
        """Test handling of edge cases in task complexity analysis."""
        # Empty task
        complexity = analyze_task_complexity("")
        assert complexity in TaskComplexity, "Should return valid complexity for empty task"
        
        # Very long task
        long_task = "analyze " * 1000
        complexity = analyze_task_complexity(long_task)
        assert complexity in TaskComplexity, "Should handle very long tasks"
        
        # Task with special characters
        special_task = "What is 2+2? @#$%^&*()[]{}|\\:;\"'<>,.?/"
        complexity = analyze_task_complexity(special_task)
        assert complexity in TaskComplexity, "Should handle special characters"
    
    def test_model_selection_with_invalid_preferences(self):
        """Test model selection with invalid or conflicting preferences."""
        task = "Test task"
        
        # Conflicting priorities
        config = select_model_for_task(
            task=task,
            preferences={
                "priority_speed": True,
                "priority_cost": True  # Conflicting with speed
            }
        )
        assert config is not None, "Should handle conflicting preferences gracefully"
        
        # Invalid schema type
        config = select_model_for_task(
            task=task,
            preferences={
                "enable_structured_output": True,
                "output_schema_type": "invalid_type"
            }
        )
        assert config is not None, "Should handle invalid schema types"
    
    def test_agent_creation_with_invalid_config(self):
        """Test agent creation with invalid configurations."""
        # Test with None model
        agent = self.factory.create_llm_agent(
            role=LLMRole.GENERALIST,
            model=None  # Should fall back to auto-selection
        )
        assert agent is not None, "Should create agent with None model"
        
        # Test with invalid custom config
        agent = self.factory.create_llm_agent(
            role=LLMRole.GENERALIST,
            custom_config={"invalid_field": "invalid_value"}
        )
        assert agent is not None, "Should handle invalid config fields gracefully"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
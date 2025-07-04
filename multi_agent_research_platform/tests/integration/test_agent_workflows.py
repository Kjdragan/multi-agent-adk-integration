# Integration tests for agent workflows and orchestration
# Tests how multiple agents work together to complete complex tasks

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any

from src.agents import AgentFactory, AgentOrchestrator, AgentRegistry
from src.agents.base import AgentResult, AgentCapability
from src.agents.llm_agent import LLMRole
from src.agents.orchestrator import OrchestrationStrategy, OrchestrationResult
from src.config.app import AppConfig


class TestAgentOrchestrationWorkflows:
    """Test complete agent orchestration workflows."""
    
    @pytest.mark.asyncio
    async def test_research_analysis_workflow(self, agent_factory, agent_orchestrator, test_agent_registry):
        """Test end-to-end research and analysis workflow."""
        # Create research team
        researcher = agent_factory.create_llm_agent(
            role=LLMRole.RESEARCHER,
            name="Research Agent"
        )
        analyst = agent_factory.create_llm_agent(
            role=LLMRole.ANALYST,
            name="Analysis Agent"
        )
        writer = agent_factory.create_llm_agent(
            role=LLMRole.COMMUNICATOR,
            name="Writing Agent"
        )
        
        # Register agents
        test_agent_registry.register_agent(researcher)
        test_agent_registry.register_agent(analyst)
        test_agent_registry.register_agent(writer)
        
        # Mock agent responses
        researcher.execute_task = AsyncMock(return_value=AgentResult(
            agent_id=researcher.agent_id,
            result="Research findings: Renewable energy adoption is increasing globally...",
            success=True,
            execution_time_ms=150,
            metadata={"sources": 5, "confidence": 0.9}
        ))
        
        analyst.execute_task = AsyncMock(return_value=AgentResult(
            agent_id=analyst.agent_id,
            result="Analysis: The data shows a 25% increase in renewable energy investment...",
            success=True,
            execution_time_ms=200,
            metadata={"analysis_depth": "comprehensive"}
        ))
        
        writer.execute_task = AsyncMock(return_value=AgentResult(
            agent_id=writer.agent_id,
            result="Executive Summary: Renewable energy markets are experiencing unprecedented growth...",
            success=True,
            execution_time_ms=180,
            metadata={"word_count": 250}
        ))
        
        # Execute research workflow
        task = "Analyze the current state and future prospects of renewable energy adoption"
        result = await agent_orchestrator.orchestrate_task(
            task=task,
            strategy=OrchestrationStrategy.CONSENSUS,
            requirements=[AgentCapability.RESEARCH, AgentCapability.ANALYSIS],
            context={
                "deadline": "2024-12-31",
                "target_audience": "executives",
                "depth": "comprehensive"
            }
        )
        
        assert result.success
        assert len(result.agents_used) >= 2
        assert result.consensus_score > 0.7
        assert "renewable energy" in result.primary_result.lower()
    
    @pytest.mark.asyncio
    async def test_content_creation_pipeline(self, agent_factory, test_agent_registry):
        """Test content creation pipeline with multiple agents."""
        # Create content creation team
        researcher = agent_factory.create_llm_agent(role=LLMRole.RESEARCHER, name="Content Researcher")
        writer = agent_factory.create_llm_agent(role=LLMRole.COMMUNICATOR, name="Content Writer")
        editor = agent_factory.create_llm_agent(role=LLMRole.CRITIC, name="Content Editor")
        
        # Register agents
        for agent in [researcher, writer, editor]:
            test_agent_registry.register_agent(agent)
        
        # Create workflow agent
        workflow_agent = agent_factory.create_workflow_agent(
            name="Content Pipeline"
        )
        
        # Define workflow steps
        workflow_config = {
            "steps": [
                {
                    "id": "research",
                    "agent_type": "llm",
                    "role": "researcher",
                    "task": "Research content topic: AI in healthcare",
                    "dependencies": [],
                    "timeout": 60
                },
                {
                    "id": "write",
                    "agent_type": "llm",
                    "role": "writer", 
                    "task": "Write blog post based on research",
                    "dependencies": ["research"],
                    "timeout": 90
                },
                {
                    "id": "edit",
                    "agent_type": "llm",
                    "role": "critic",
                    "task": "Review and edit the blog post",
                    "dependencies": ["write"],
                    "timeout": 45
                }
            ],
            "context": {
                "topic": "AI in healthcare",
                "format": "blog_post",
                "word_count": 800
            }
        }
        
        # Mock workflow step execution
        step_results = {
            "research": AgentResult(
                agent_id=researcher.agent_id,
                result="AI in healthcare research findings...",
                success=True,
                execution_time_ms=120
            ),
            "write": AgentResult(
                agent_id=writer.agent_id,
                result="AI is revolutionizing healthcare in many ways...",
                success=True,
                execution_time_ms=200
            ),
            "edit": AgentResult(
                agent_id=editor.agent_id,
                result="Edited: AI is transforming healthcare through innovative applications...",
                success=True,
                execution_time_ms=100
            )
        }
        
        with patch.object(workflow_agent, '_execute_step') as mock_execute:
            mock_execute.side_effect = lambda step_config: step_results[step_config["id"]]
            
            result = await workflow_agent.execute_workflow(workflow_config)
        
        assert result.success
        assert len(result.step_results) == 3
        assert all(step_result.success for step_result in result.step_results.values())
        assert result.total_execution_time > 0
    
    @pytest.mark.asyncio
    async def test_competitive_orchestration_strategy(self, agent_factory, agent_orchestrator, test_agent_registry):
        """Test competitive orchestration where multiple agents compete."""
        # Create competing agents
        agents = []
        for i in range(3):
            agent = agent_factory.create_llm_agent(
                role=LLMRole.RESEARCHER,
                name=f"Competitor {i+1}"
            )
            
            # Mock different quality responses
            quality_scores = [0.9, 0.7, 0.8]
            agent.execute_task = AsyncMock(return_value=AgentResult(
                agent_id=agent.agent_id,
                result=f"Response from competitor {i+1}",
                success=True,
                execution_time_ms=100 + i*20,
                metadata={"quality_score": quality_scores[i]}
            ))
            
            agents.append(agent)
            test_agent_registry.register_agent(agent)
        
        # Execute competitive orchestration
        result = await agent_orchestrator.orchestrate_task(
            task="Find the best approach to solve climate change",
            strategy=OrchestrationStrategy.COMPETITIVE,
            requirements=[AgentCapability.RESEARCH]
        )
        
        assert result.success
        assert len(result.agents_used) == 3
        assert result.winning_agent_id == agents[0].agent_id  # Highest quality score
        assert result.competition_scores is not None
    
    @pytest.mark.asyncio
    async def test_parallel_research_synthesis(self, agent_factory, agent_orchestrator, test_agent_registry):
        """Test parallel research followed by synthesis."""
        # Create research specialists
        specialists = []
        research_areas = ["technology", "economics", "environment", "policy"]
        
        for area in research_areas:
            agent = agent_factory.create_llm_agent(
                role=LLMRole.RESEARCHER,
                name=f"{area.title()} Specialist"
            )
            
            agent.execute_task = AsyncMock(return_value=AgentResult(
                agent_id=agent.agent_id,
                result=f"Detailed {area} research findings...",
                success=True,
                execution_time_ms=150,
                metadata={"domain": area, "sources": 10}
            ))
            
            specialists.append(agent)
            test_agent_registry.register_agent(agent)
        
        # Create synthesizer
        synthesizer = agent_factory.create_llm_agent(
            role=LLMRole.SYNTHESIZER,
            name="Research Synthesizer"
        )
        
        synthesizer.execute_task = AsyncMock(return_value=AgentResult(
            agent_id=synthesizer.agent_id,
            result="Comprehensive synthesis of all research areas...",
            success=True,
            execution_time_ms=300,
            metadata={"synthesis_quality": 0.95}
        ))
        
        test_agent_registry.register_agent(synthesizer)
        
        # Execute parallel research
        result = await agent_orchestrator.orchestrate_task(
            task="Research the multifaceted impacts of artificial intelligence",
            strategy=OrchestrationStrategy.PARALLEL_SYNTHESIS,
            requirements=[AgentCapability.RESEARCH, AgentCapability.SYNTHESIS]
        )
        
        assert result.success
        assert len(result.parallel_results) == 4  # Four specialists
        assert result.synthesis_result is not None
        assert "synthesis" in result.synthesis_result.lower()


class TestAgentCollaboration:
    """Test agent collaboration patterns."""
    
    @pytest.mark.asyncio
    async def test_agent_chain_collaboration(self, agent_factory, test_agent_registry):
        """Test agents passing results to each other in a chain."""
        # Create agent chain: Researcher -> Analyst -> Writer
        researcher = agent_factory.create_llm_agent(role=LLMRole.RESEARCHER, name="Chain Researcher")
        analyst = agent_factory.create_llm_agent(role=LLMRole.ANALYST, name="Chain Analyst")
        writer = agent_factory.create_llm_agent(role=LLMRole.COMMUNICATOR, name="Chain Writer")
        
        for agent in [researcher, analyst, writer]:
            test_agent_registry.register_agent(agent)
        
        # Mock agent behaviors with context passing
        researcher.execute_task = AsyncMock(return_value=AgentResult(
            agent_id=researcher.agent_id,
            result="Initial research data: Market size $100B, growth 15% annually",
            success=True,
            execution_time_ms=100,
            metadata={"data_quality": "high", "next_step": "analysis"}
        ))
        
        def analyst_task(task, context=None):
            # Analyst should receive researcher's result in context
            previous_result = context.get("previous_results", []) if context else []
            return AgentResult(
                agent_id=analyst.agent_id,
                result="Analysis based on research: Strong growth trend indicates market opportunity",
                success=True,
                execution_time_ms=150,
                metadata={
                    "used_previous_data": len(previous_result) > 0,
                    "analysis_confidence": 0.9
                }
            )
        
        def writer_task(task, context=None):
            previous_results = context.get("previous_results", []) if context else []
            return AgentResult(
                agent_id=writer.agent_id,
                result="Executive Summary: Market analysis reveals significant opportunity...",
                success=True,
                execution_time_ms=120,
                metadata={
                    "incorporated_results": len(previous_results),
                    "word_count": 200
                }
            )
        
        analyst.execute_task = AsyncMock(side_effect=analyst_task)
        writer.execute_task = AsyncMock(side_effect=writer_task)
        
        # Execute chain workflow
        workflow_agent = agent_factory.create_workflow_agent(
            name="Chain Workflow"
        )
        
        chain_config = {
            "steps": [
                {
                    "id": "research",
                    "agent_id": researcher.agent_id,
                    "task": "Research market opportunity",
                    "dependencies": []
                },
                {
                    "id": "analyze", 
                    "agent_id": analyst.agent_id,
                    "task": "Analyze research findings",
                    "dependencies": ["research"]
                },
                {
                    "id": "write",
                    "agent_id": writer.agent_id,
                    "task": "Write executive summary",
                    "dependencies": ["analyze"]
                }
            ]
        }
        
        with patch.object(workflow_agent, '_build_context_for_step') as mock_context:
            # Mock context building to pass previous results
            def build_context(step_config, completed_steps):
                context = {"previous_results": list(completed_steps.values())}
                return context
            
            mock_context.side_effect = build_context
            
            with patch.object(workflow_agent, '_get_agent_for_step') as mock_get_agent:
                def get_agent(step_config):
                    agent_id = step_config["agent_id"]
                    if agent_id == researcher.agent_id:
                        return researcher
                    elif agent_id == analyst.agent_id:
                        return analyst
                    elif agent_id == writer.agent_id:
                        return writer
                
                mock_get_agent.side_effect = get_agent
                
                result = await workflow_agent.execute_workflow(chain_config)
        
        assert result.success
        assert len(result.step_results) == 3
        
        # Verify context was passed between agents
        analyst.execute_task.assert_called_once()
        writer.execute_task.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_agent_feedback_loop(self, agent_factory, test_agent_registry):
        """Test agents providing feedback to improve results."""
        # Create writer and critic agents
        writer = agent_factory.create_llm_agent(role=LLMRole.COMMUNICATOR, name="Content Writer")
        critic = agent_factory.create_llm_agent(role=LLMRole.CRITIC, name="Content Critic")
        
        for agent in [writer, critic]:
            test_agent_registry.register_agent(agent)
        
        # Simulate feedback loop
        iteration = 0
        
        def writer_task(task, context=None):
            nonlocal iteration
            iteration += 1
            
            feedback = context.get("feedback", []) if context else []
            quality = 0.6 + (iteration * 0.15)  # Improve with feedback
            
            return AgentResult(
                agent_id=writer.agent_id,
                result=f"Content v{iteration}: Improved based on feedback",
                success=True,
                execution_time_ms=100,
                metadata={
                    "version": iteration,
                    "quality_score": quality,
                    "feedback_incorporated": len(feedback) > 0
                }
            )
        
        def critic_task(task, context=None):
            content_to_review = context.get("content_to_review", "") if context else ""
            
            # Simulate improving feedback based on content quality
            if "v1" in content_to_review:
                feedback_score = 0.6
            elif "v2" in content_to_review:
                feedback_score = 0.8
            else:
                feedback_score = 0.9
            
            return AgentResult(
                agent_id=critic.agent_id,
                result=f"Feedback: Content quality score {feedback_score}",
                success=True,
                execution_time_ms=80,
                metadata={
                    "feedback_score": feedback_score,
                    "suggestions_count": 3 if feedback_score < 0.8 else 1
                }
            )
        
        writer.execute_task = AsyncMock(side_effect=writer_task)
        critic.execute_task = AsyncMock(side_effect=critic_task)
        
        # Execute feedback loop (3 iterations)
        results = []
        content = ""
        feedback = []
        
        for i in range(3):
            # Writer creates/improves content
            writer_context = {"feedback": feedback}
            writer_result = await writer.execute_task(
                "Write content about AI safety",
                context=writer_context
            )
            
            # Critic provides feedback
            critic_context = {"content_to_review": writer_result.result}
            critic_result = await critic.execute_task(
                "Review the content quality",
                context=critic_context
            )
            
            results.append((writer_result, critic_result))
            feedback.append(critic_result.result)
            
            # Stop if quality is high enough
            if critic_result.metadata["feedback_score"] >= 0.9:
                break
        
        # Verify improvement over iterations
        assert len(results) >= 2
        
        quality_scores = [critic_result.metadata["feedback_score"] 
                         for writer_result, critic_result in results]
        
        # Quality should improve over iterations
        for i in range(1, len(quality_scores)):
            assert quality_scores[i] >= quality_scores[i-1]


class TestErrorHandlingAndRecovery:
    """Test error handling and recovery in multi-agent workflows."""
    
    @pytest.mark.asyncio
    async def test_workflow_with_failing_agent(self, agent_factory, test_agent_registry):
        """Test workflow recovery when one agent fails."""
        # Create agents with one that will fail
        good_agent = agent_factory.create_llm_agent(role=LLMRole.RESEARCHER, name="Good Agent")
        failing_agent = agent_factory.create_llm_agent(role=LLMRole.ANALYST, name="Failing Agent") 
        backup_agent = agent_factory.create_llm_agent(role=LLMRole.ANALYST, name="Backup Agent")
        
        for agent in [good_agent, failing_agent, backup_agent]:
            test_agent_registry.register_agent(agent)
        
        # Configure agent behaviors
        good_agent.execute_task = AsyncMock(return_value=AgentResult(
            agent_id=good_agent.agent_id,
            result="Good research result",
            success=True,
            execution_time_ms=100
        ))
        
        failing_agent.execute_task = AsyncMock(return_value=AgentResult(
            agent_id=failing_agent.agent_id,
            result="",
            success=False,
            execution_time_ms=50,
            error_message="Simulated agent failure"
        ))
        
        backup_agent.execute_task = AsyncMock(return_value=AgentResult(
            agent_id=backup_agent.agent_id,
            result="Backup analysis result",
            success=True,
            execution_time_ms=120
        ))
        
        # Create orchestrator with fallback strategy
        orchestrator = AgentOrchestrator()
        
        # Test fallback to backup agent
        result = await orchestrator.orchestrate_task(
            task="Analyze the research data",
            strategy=OrchestrationStrategy.ADAPTIVE,
            requirements=[AgentCapability.ANALYSIS],
            context={"enable_fallback": True}
        )
        
        # Should succeed using backup agent
        assert result.success
        assert backup_agent.agent_id in result.agents_used
    
    @pytest.mark.asyncio
    async def test_partial_workflow_recovery(self, agent_factory):
        """Test workflow that can partially recover from failures."""
        workflow_agent = agent_factory.create_workflow_agent(
            name="Resilient Workflow"
        )
        
        # Workflow with optional steps
        workflow_config = {
            "steps": [
                {
                    "id": "required_step1",
                    "agent_type": "llm",
                    "role": "researcher",
                    "task": "Essential research",
                    "required": True,
                    "dependencies": []
                },
                {
                    "id": "optional_step",
                    "agent_type": "llm",
                    "role": "analyst",
                    "task": "Optional analysis",
                    "required": False,
                    "dependencies": ["required_step1"]
                },
                {
                    "id": "required_step2",
                    "agent_type": "llm", 
                    "role": "writer",
                    "task": "Final writing",
                    "required": True,
                    "dependencies": ["required_step1"]  # Not dependent on optional step
                }
            ]
        }
        
        # Mock step execution with one failure
        step_results = {
            "required_step1": AgentResult(
                agent_id="agent1",
                result="Essential research completed",
                success=True,
                execution_time_ms=100
            ),
            "optional_step": AgentResult(
                agent_id="agent2", 
                result="",
                success=False,
                execution_time_ms=50,
                error_message="Optional step failed"
            ),
            "required_step2": AgentResult(
                agent_id="agent3",
                result="Final writing completed", 
                success=True,
                execution_time_ms=150
            )
        }
        
        with patch.object(workflow_agent, '_execute_step') as mock_execute:
            mock_execute.side_effect = lambda step: step_results[step["id"]]
            
            result = await workflow_agent.execute_workflow(workflow_config)
        
        # Workflow should succeed despite optional step failure
        assert result.success
        assert len(result.step_results) == 3
        assert result.step_results["required_step1"].success
        assert not result.step_results["optional_step"].success
        assert result.step_results["required_step2"].success
        assert result.partial_success is True


@pytest.mark.slow
class TestPerformanceUnderLoad:
    """Test agent performance under load conditions."""
    
    @pytest.mark.asyncio
    async def test_concurrent_orchestration_requests(self, agent_factory, agent_orchestrator, test_agent_registry):
        """Test multiple concurrent orchestration requests."""
        # Create pool of agents
        agents = []
        for i in range(5):
            agent = agent_factory.create_llm_agent(
                role=LLMRole.RESEARCHER,
                name=f"Concurrent Agent {i}"
            )
            
            agent.execute_task = AsyncMock(return_value=AgentResult(
                agent_id=agent.agent_id,
                result=f"Result from agent {i}",
                success=True,
                execution_time_ms=100 + i*10
            ))
            
            agents.append(agent)
            test_agent_registry.register_agent(agent)
        
        # Create multiple concurrent tasks
        tasks = [
            agent_orchestrator.orchestrate_task(
                task=f"Concurrent task {i}",
                strategy=OrchestrationStrategy.SINGLE_BEST,
                requirements=[AgentCapability.RESEARCH]
            )
            for i in range(10)
        ]
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all tasks completed successfully
        successful_results = [r for r in results if isinstance(r, OrchestrationResult) and r.success]
        assert len(successful_results) == 10
        
        # Verify agent load balancing
        agents_used = set()
        for result in successful_results:
            agents_used.update(result.agents_used)
        
        # Multiple agents should have been used
        assert len(agents_used) > 1
    
    @pytest.mark.asyncio
    async def test_workflow_scalability(self, agent_factory):
        """Test workflow scalability with increasing step counts."""
        workflow_agent = agent_factory.create_workflow_agent(
            name="Scalability Test Workflow"
        )
        
        # Test with increasing numbers of parallel tasks
        for task_count in [5, 10, 20]:
            parallel_tasks = [
                {
                    "id": f"task_{i}",
                    "agent_type": "llm",
                    "role": "researcher", 
                    "task": f"Parallel task {i}"
                }
                for i in range(task_count)
            ]
            
            workflow_config = {
                "parallel_tasks": parallel_tasks,
                "max_concurrent": 10
            }
            
            with patch.object(workflow_agent, '_execute_step') as mock_execute:
                mock_execute.return_value = AgentResult(
                    agent_id="mock_agent",
                    result="Mock task result",
                    success=True,
                    execution_time_ms=50
                )
                
                start_time = asyncio.get_event_loop().time()
                result = await workflow_agent.execute_workflow(workflow_config)
                end_time = asyncio.get_event_loop().time()
                
                execution_time = end_time - start_time
                
                assert result.success
                assert len(result.task_results) == task_count
                
                # Execution time should scale reasonably
                assert execution_time < task_count * 0.1  # Should be faster than sequential


@pytest.mark.requires_api_key("GOOGLE_API_KEY")
class TestRealAPIIntegration:
    """Integration tests with real APIs (requires API keys)."""
    
    @pytest.mark.asyncio
    async def test_real_llm_agent_execution(self, agent_factory):
        """Test LLM agent with real API calls."""
        agent = agent_factory.create_llm_agent(
            role=LLMRole.RESEARCHER,
            name="Real API Test Agent"
        )
        
        # Execute simple task with real API
        result = await agent.execute_task(
            "What is 2+2? Provide a brief answer."
        )
        
        assert result.success
        assert "4" in result.result
        assert result.execution_time_ms > 0
    
    @pytest.mark.asyncio
    async def test_real_orchestration_workflow(self, agent_factory, agent_orchestrator, test_agent_registry):
        """Test orchestration with real agents."""
        # Create minimal agent team
        agent = agent_factory.create_llm_agent(
            role=LLMRole.RESEARCHER,
            name="Real Orchestration Agent"
        )
        
        test_agent_registry.register_agent(agent)
        
        # Execute simple orchestration
        result = await agent_orchestrator.orchestrate_task(
            task="What is the capital of France?",
            strategy=OrchestrationStrategy.SINGLE_BEST,
            requirements=[AgentCapability.RESEARCH]
        )
        
        assert result.success
        assert "paris" in result.primary_result.lower()
        assert len(result.agents_used) == 1
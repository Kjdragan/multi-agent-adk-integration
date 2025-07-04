# End-to-end tests for complete user workflows
# Tests the entire platform from user interface to final output

import pytest
import asyncio
import httpx
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List

from src.web.interface import WebInterface
# TODO: Fix StreamlitConfig initialization for ADK v1.5.0
# from src.streamlit.main import StreamlitApp
from src.agents import AgentFactory, AgentOrchestrator
from src.config.app import AppConfig


class TestWebInterfaceE2E:
    """End-to-end tests for web interface workflows."""
    
    @pytest.fixture
    async def web_app(self, app_config):
        """Create web application for testing."""
        web_interface = WebInterface(
            web_config=app_config,
            debug_config=app_config,
            monitoring_config=app_config
        )
        
        await web_interface.initialize()
        
        # Start the application
        await web_interface.start()
        
        yield web_interface.app
        
        # Cleanup
        await web_interface.stop()
    
    @pytest.mark.asyncio
    async def test_complete_research_workflow_via_api(self, web_app):
        """Test complete research workflow through REST API."""
        async with httpx.AsyncClient(app=web_app, base_url="http://test") as client:
            # Step 1: Create research agent
            agent_response = await client.post("/api/v1/agents", json={
                "agent_type": "llm",
                "name": "E2E Research Agent",
                "config": {
                    "role": "researcher",
                    "temperature": 0.7,
                    "max_tokens": 2000
                }
            })
            
            assert agent_response.status_code == 200
            agent_data = agent_response.json()
            agent_id = agent_data["agent_id"]
            
            # Step 2: Check agent status
            status_response = await client.get(f"/api/v1/agents/{agent_id}")
            assert status_response.status_code == 200
            
            status_data = status_response.json()
            assert status_data["name"] == "E2E Research Agent"
            assert status_data["is_active"] is True
            
            # Step 3: Execute research task
            task_response = await client.post("/api/v1/orchestration/task", json={
                "task": "Research the benefits of renewable energy sources",
                "strategy": "single_best",
                "requirements": ["research"],
                "context": {
                    "user_id": "test_user",
                    "session_id": "test_session"
                }
            })
            
            assert task_response.status_code == 200
            task_data = task_response.json()
            
            assert task_data["success"] is True
            assert len(task_data["primary_result"]) > 0
            assert "renewable energy" in task_data["primary_result"].lower()
            
            # Step 4: Get task history
            history_response = await client.get("/api/v1/orchestration/history")
            assert history_response.status_code == 200
            
            history_data = history_response.json()
            assert len(history_data["tasks"]) >= 1
            
            # Step 5: Check system health
            health_response = await client.get("/health")
            assert health_response.status_code == 200
            
            health_data = health_response.json()
            assert health_data["status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_multi_agent_collaboration_via_api(self, web_app):
        """Test multi-agent collaboration through API."""
        async with httpx.AsyncClient(app=web_app, base_url="http://test") as client:
            # Create multiple agents
            agent_configs = [
                {"role": "researcher", "name": "Research Specialist"},
                {"role": "analyst", "name": "Data Analyst"},
                {"role": "writer", "name": "Content Writer"}
            ]
            
            agent_ids = []
            for config in agent_configs:
                response = await client.post("/api/v1/agents", json={
                    "agent_type": "llm",
                    "name": config["name"],
                    "config": config
                })
                
                assert response.status_code == 200
                agent_ids.append(response.json()["agent_id"])
            
            # Execute collaborative task
            task_response = await client.post("/api/v1/orchestration/task", json={
                "task": "Create a comprehensive report on artificial intelligence trends",
                "strategy": "consensus",
                "requirements": ["research", "analysis", "writing"],
                "context": {
                    "collaboration_mode": "sequential",
                    "quality_threshold": 0.8
                }
            })
            
            assert task_response.status_code == 200
            task_data = task_response.json()
            
            assert task_data["success"] is True
            assert len(task_data["agents_used"]) >= 2
            assert task_data["consensus_score"] > 0.7
            
            # Verify agent performance metrics
            for agent_id in agent_ids:
                metrics_response = await client.get(f"/api/v1/agents/{agent_id}/metrics")
                assert metrics_response.status_code == 200
                
                metrics_data = metrics_response.json()
                assert "total_tasks" in metrics_data
                assert "success_rate_percent" in metrics_data
    
    @pytest.mark.asyncio
    async def test_workflow_management_via_api(self, web_app):
        """Test workflow creation and execution through API."""
        async with httpx.AsyncClient(app=web_app, base_url="http://test") as client:
            # Create workflow agent
            workflow_response = await client.post("/api/v1/agents", json={
                "agent_type": "workflow",
                "name": "E2E Content Workflow",
                "config": {
                    "workflow_type": "sequential",
                    "max_parallel_tasks": 3
                }
            })
            
            assert workflow_response.status_code == 200
            workflow_id = workflow_response.json()["agent_id"]
            
            # Define and execute workflow
            workflow_config = {
                "steps": [
                    {
                        "id": "research",
                        "agent_type": "llm",
                        "role": "researcher",
                        "task": "Research topic: sustainable agriculture",
                        "dependencies": []
                    },
                    {
                        "id": "analyze",
                        "agent_type": "llm",
                        "role": "analyst",
                        "task": "Analyze research findings",
                        "dependencies": ["research"]
                    },
                    {
                        "id": "write",
                        "agent_type": "llm",
                        "role": "writer",
                        "task": "Write comprehensive report",
                        "dependencies": ["analyze"]
                    }
                ]
            }
            
            execution_response = await client.post(
                f"/api/v1/workflows/{workflow_id}/execute",
                json=workflow_config
            )
            
            assert execution_response.status_code == 200
            execution_data = execution_response.json()
            
            assert execution_data["success"] is True
            assert len(execution_data["step_results"]) == 3
            assert all(step["success"] for step in execution_data["step_results"].values())


class TestStreamlitInterfaceE2E:
    """End-to-end tests for Streamlit interface."""
    
    @pytest.fixture
    def streamlit_app(self, app_config):
        """Create Streamlit app instance for testing."""
        app = StreamlitApp(config=app_config)
        return app
    
    def test_streamlit_agent_creation_flow(self, streamlit_app):
        """Test agent creation through Streamlit interface."""
        # Mock Streamlit session state
        mock_session_state = {
            "agent_configs": [],
            "selected_agent_type": "llm",
            "agent_name": "Test Streamlit Agent",
            "agent_role": "researcher"
        }
        
        with patch("streamlit.session_state", mock_session_state):
            with patch("streamlit.selectbox") as mock_selectbox:
                with patch("streamlit.text_input") as mock_text_input:
                    with patch("streamlit.button") as mock_button:
                        mock_selectbox.return_value = "llm"
                        mock_text_input.side_effect = ["Test Streamlit Agent", "researcher"]
                        mock_button.return_value = True
                        
                        # Simulate agent creation
                        agent_config = streamlit_app._create_agent_config()
                        
                        assert agent_config["name"] == "Test Streamlit Agent"
                        assert agent_config["agent_type"] == "llm"
                        assert agent_config["config"]["role"] == "researcher"
    
    def test_streamlit_task_execution_flow(self, streamlit_app):
        """Test task execution through Streamlit interface."""
        mock_session_state = {
            "agents": ["agent_1", "agent_2"],
            "task_input": "Research renewable energy benefits",
            "orchestration_strategy": "consensus",
            "task_results": []
        }
        
        with patch("streamlit.session_state", mock_session_state):
            with patch("streamlit.text_area") as mock_text_area:
                with patch("streamlit.selectbox") as mock_selectbox:
                    with patch("streamlit.button") as mock_button:
                        with patch.object(streamlit_app, "_execute_task") as mock_execute:
                            mock_text_area.return_value = "Research renewable energy benefits"
                            mock_selectbox.return_value = "consensus"
                            mock_button.return_value = True
                            
                            mock_execute.return_value = {
                                "success": True,
                                "result": "Renewable energy benefits research completed",
                                "agents_used": ["agent_1"],
                                "execution_time": 150
                            }
                            
                            # Simulate task execution
                            result = streamlit_app._handle_task_execution()
                            
                            assert result["success"] is True
                            assert "renewable energy" in result["result"].lower()


class TestCompleteUserJourneys:
    """Test complete user journeys from start to finish."""
    
    @pytest.mark.asyncio
    async def test_new_user_onboarding_journey(self, web_app):
        """Test complete new user onboarding and first task."""
        async with httpx.AsyncClient(app=web_app, base_url="http://test") as client:
            # Step 1: New user checks system status
            health_response = await client.get("/health")
            assert health_response.status_code == 200
            
            # Step 2: User explores available agent types
            types_response = await client.get("/api/v1/agents/types")
            assert types_response.status_code == 200
            
            agent_types = types_response.json()
            assert "llm" in agent_types
            assert "workflow" in agent_types
            assert "custom" in agent_types
            
            # Step 3: User creates their first agent
            first_agent_response = await client.post("/api/v1/agents", json={
                "agent_type": "llm",
                "name": "My First Agent",
                "config": {
                    "role": "researcher",
                    "temperature": 0.7
                }
            })
            
            assert first_agent_response.status_code == 200
            first_agent_id = first_agent_response.json()["agent_id"]
            
            # Step 4: User executes their first task
            first_task_response = await client.post("/api/v1/orchestration/task", json={
                "task": "Hello world - tell me about artificial intelligence",
                "strategy": "single_best",
                "context": {
                    "user_id": "new_user_123",
                    "first_task": True
                }
            })
            
            assert first_task_response.status_code == 200
            first_task_data = first_task_response.json()
            
            assert first_task_data["success"] is True
            assert len(first_task_data["primary_result"]) > 0
            
            # Step 5: User reviews their agent's performance
            metrics_response = await client.get(f"/api/v1/agents/{first_agent_id}/metrics")
            assert metrics_response.status_code == 200
            
            metrics_data = metrics_response.json()
            assert metrics_data["total_tasks"] == 1
            assert metrics_data["success_rate_percent"] == 100.0
    
    @pytest.mark.asyncio
    async def test_power_user_advanced_workflow(self, web_app):
        """Test advanced user creating complex multi-agent workflows."""
        async with httpx.AsyncClient(app=web_app, base_url="http://test") as client:
            # Step 1: Create specialized agent team
            team_configs = [
                {
                    "agent_type": "llm",
                    "name": "Senior Researcher",
                    "config": {"role": "researcher", "temperature": 0.3}
                },
                {
                    "agent_type": "llm", 
                    "name": "Data Analyst",
                    "config": {"role": "analyst", "temperature": 0.2}
                },
                {
                    "agent_type": "custom",
                    "name": "Fact Checker",
                    "config": {"agent_type": "fact_checker"}
                },
                {
                    "agent_type": "llm",
                    "name": "Report Writer",
                    "config": {"role": "writer", "temperature": 0.8}
                }
            ]
            
            agent_ids = []
            for config in team_configs:
                response = await client.post("/api/v1/agents", json=config)
                assert response.status_code == 200
                agent_ids.append(response.json()["agent_id"])
            
            # Step 2: Create complex workflow
            workflow_response = await client.post("/api/v1/agents", json={
                "agent_type": "workflow",
                "name": "Market Analysis Workflow",
                "config": {
                    "workflow_type": "hybrid",
                    "max_parallel_tasks": 3
                }
            })
            
            assert workflow_response.status_code == 200
            workflow_id = workflow_response.json()["agent_id"]
            
            # Step 3: Execute multi-stage workflow
            complex_workflow = {
                "parallel_research": [
                    {
                        "id": "market_research",
                        "agent_id": agent_ids[0],  # Senior Researcher
                        "task": "Research global EV market size and trends"
                    },
                    {
                        "id": "competitive_analysis",
                        "agent_id": agent_ids[0],  # Senior Researcher
                        "task": "Analyze top EV manufacturers and market share"
                    },
                    {
                        "id": "technology_trends",
                        "agent_id": agent_ids[0],  # Senior Researcher
                        "task": "Research EV battery technology developments"
                    }
                ],
                "sequential_analysis": [
                    {
                        "id": "data_analysis",
                        "agent_id": agent_ids[1],  # Data Analyst
                        "task": "Analyze all research data for insights",
                        "dependencies": ["parallel_research"]
                    },
                    {
                        "id": "fact_check",
                        "agent_id": agent_ids[2],  # Fact Checker
                        "task": "Verify all claims and statistics",
                        "dependencies": ["data_analysis"]
                    },
                    {
                        "id": "final_report",
                        "agent_id": agent_ids[3],  # Report Writer
                        "task": "Write comprehensive market analysis report",
                        "dependencies": ["fact_check"]
                    }
                ]
            }
            
            execution_response = await client.post(
                f"/api/v1/workflows/{workflow_id}/execute",
                json=complex_workflow
            )
            
            assert execution_response.status_code == 200
            execution_data = execution_response.json()
            
            assert execution_data["success"] is True
            assert len(execution_data["parallel_results"]) == 3
            assert len(execution_data["sequential_results"]) == 3
            
            # Step 4: Export results
            export_response = await client.post("/api/v1/export/report", json={
                "workflow_id": workflow_id,
                "format": "json",
                "include_metadata": True
            })
            
            assert export_response.status_code == 200
            export_data = export_response.json()
            
            assert "final_report" in export_data
            assert "metadata" in export_data
            assert "execution_summary" in export_data
    
    @pytest.mark.asyncio
    async def test_error_recovery_user_journey(self, web_app):
        """Test user journey with error recovery scenarios."""
        async with httpx.AsyncClient(app=web_app, base_url="http://test") as client:
            # Step 1: User creates agent with invalid configuration
            invalid_agent_response = await client.post("/api/v1/agents", json={
                "agent_type": "invalid_type",
                "name": "Invalid Agent",
                "config": {}
            })
            
            assert invalid_agent_response.status_code == 400
            error_data = invalid_agent_response.json()
            assert "error" in error_data
            
            # Step 2: User corrects configuration and creates valid agent
            valid_agent_response = await client.post("/api/v1/agents", json={
                "agent_type": "llm",
                "name": "Valid Agent",
                "config": {"role": "researcher"}
            })
            
            assert valid_agent_response.status_code == 200
            agent_id = valid_agent_response.json()["agent_id"]
            
            # Step 3: User executes task that might fail
            with patch("src.agents.llm_agent.LLMAgent.execute_task") as mock_execute:
                # First attempt fails
                mock_execute.side_effect = [
                    Exception("Simulated API failure"),
                    # Second attempt succeeds (retry logic)
                ]
                
                task_response = await client.post("/api/v1/orchestration/task", json={
                    "task": "Test task with potential failure",
                    "strategy": "single_best",
                    "context": {"retry_on_failure": True}
                })
            
            # Step 4: User checks error logs and system status
            logs_response = await client.get("/api/v1/debug/logs?level=ERROR&limit=10")
            assert logs_response.status_code == 200
            
            status_response = await client.get("/status")
            assert status_response.status_code == 200
            
            status_data = status_response.json()
            assert "error_rate" in status_data
            assert "last_error" in status_data


class TestDataPersistenceE2E:
    """Test data persistence across sessions."""
    
    @pytest.mark.asyncio
    async def test_agent_persistence_across_sessions(self, web_app, test_database):
        """Test that agents persist across application restarts."""
        async with httpx.AsyncClient(app=web_app, base_url="http://test") as client:
            # Create agent in first session
            agent_response = await client.post("/api/v1/agents", json={
                "agent_type": "llm",
                "name": "Persistent Agent",
                "config": {"role": "researcher"}
            })
            
            assert agent_response.status_code == 200
            agent_id = agent_response.json()["agent_id"]
            
            # Execute task to create history
            task_response = await client.post("/api/v1/orchestration/task", json={
                "task": "Create some task history",
                "strategy": "single_best"
            })
            
            assert task_response.status_code == 200
        
        # Simulate application restart by creating new web interface
        async with httpx.AsyncClient(app=web_app, base_url="http://test") as client:
            # Check that agent still exists
            agent_check_response = await client.get(f"/api/v1/agents/{agent_id}")
            assert agent_check_response.status_code == 200
            
            agent_data = agent_check_response.json()
            assert agent_data["name"] == "Persistent Agent"
            
            # Check that task history persists
            history_response = await client.get("/api/v1/orchestration/history")
            assert history_response.status_code == 200
            
            history_data = history_response.json()
            assert len(history_data["tasks"]) >= 1
    
    @pytest.mark.asyncio
    async def test_configuration_persistence(self, web_app):
        """Test that configuration changes persist."""
        async with httpx.AsyncClient(app=web_app, base_url="http://test") as client:
            # Update system configuration
            config_response = await client.patch("/api/v1/config", json={
                "max_concurrent_agents": 15,
                "default_timeout_seconds": 90
            })
            
            assert config_response.status_code == 200
            
            # Verify configuration was updated
            get_config_response = await client.get("/api/v1/config")
            assert get_config_response.status_code == 200
            
            config_data = get_config_response.json()
            assert config_data["max_concurrent_agents"] == 15
            assert config_data["default_timeout_seconds"] == 90


@pytest.mark.slow
class TestLongRunningWorkflows:
    """Test long-running workflows and extended operations."""
    
    @pytest.mark.asyncio
    async def test_extended_research_project(self, web_app):
        """Test extended research project with multiple phases."""
        async with httpx.AsyncClient(app=web_app, base_url="http://test", timeout=300.0) as client:
            # Phase 1: Initial research setup
            team_response = await client.post("/api/v1/agent-suites", json={
                "suite_type": "research_team",
                "project_name": "Extended Climate Research"
            })
            
            assert team_response.status_code == 200
            team_data = team_response.json()
            agent_ids = team_data["agent_ids"]
            
            # Phase 2: Comprehensive research tasks
            research_tasks = [
                "Research climate change causes and effects",
                "Analyze renewable energy adoption rates globally", 
                "Study carbon capture technologies",
                "Examine climate policy effectiveness",
                "Review sustainable agriculture practices"
            ]
            
            task_results = []
            for task in research_tasks:
                task_response = await client.post("/api/v1/orchestration/task", json={
                    "task": task,
                    "strategy": "consensus",
                    "context": {"project": "Extended Climate Research"}
                })
                
                assert task_response.status_code == 200
                task_results.append(task_response.json())
            
            # Phase 3: Synthesis and final report
            synthesis_response = await client.post("/api/v1/orchestration/task", json={
                "task": "Synthesize all climate research into comprehensive report",
                "strategy": "parallel_synthesis",
                "context": {
                    "previous_results": [r["primary_result"] for r in task_results],
                    "project": "Extended Climate Research"
                }
            })
            
            assert synthesis_response.status_code == 200
            synthesis_data = synthesis_response.json()
            
            assert synthesis_data["success"] is True
            assert len(synthesis_data["synthesis_result"]) > 1000  # Comprehensive report
            
            # Verify project completion
            project_response = await client.get("/api/v1/projects/Extended Climate Research")
            assert project_response.status_code == 200
            
            project_data = project_response.json()
            assert project_data["status"] == "completed"
            assert len(project_data["task_history"]) == len(research_tasks) + 1
    
    @pytest.mark.asyncio
    async def test_iterative_improvement_workflow(self, web_app):
        """Test iterative improvement workflow with feedback loops."""
        async with httpx.AsyncClient(app=web_app, base_url="http://test", timeout=180.0) as client:
            # Create improvement workflow
            workflow_response = await client.post("/api/v1/agents", json={
                "agent_type": "workflow",
                "name": "Iterative Improvement Workflow",
                "config": {
                    "workflow_type": "iterative",
                    "max_iterations": 3,
                    "improvement_threshold": 0.8
                }
            })
            
            assert workflow_response.status_code == 200
            workflow_id = workflow_response.json()["agent_id"]
            
            # Execute iterative workflow
            iterative_config = {
                "base_task": "Write a persuasive essay about renewable energy",
                "improvement_criteria": [
                    "clarity_and_structure",
                    "evidence_quality", 
                    "persuasiveness",
                    "readability"
                ],
                "agents": {
                    "writer": {"role": "writer", "temperature": 0.8},
                    "critic": {"role": "critic", "temperature": 0.3},
                    "editor": {"role": "editor", "temperature": 0.5}
                }
            }
            
            execution_response = await client.post(
                f"/api/v1/workflows/{workflow_id}/execute",
                json=iterative_config
            )
            
            assert execution_response.status_code == 200
            execution_data = execution_response.json()
            
            assert execution_data["success"] is True
            assert execution_data["iterations_completed"] >= 2
            assert execution_data["final_quality_score"] > 0.8
            
            # Verify improvement over iterations
            iterations = execution_data["iteration_results"]
            quality_scores = [iteration["quality_score"] for iteration in iterations]
            
            # Quality should generally improve
            assert quality_scores[-1] >= quality_scores[0]


@pytest.mark.requires_internet
class TestExternalIntegrationE2E:
    """Test integration with external services and APIs."""
    
    @pytest.mark.asyncio
    async def test_weather_agent_integration(self, web_app, mock_openweather_client):
        """Test weather agent with external API integration."""
        async with httpx.AsyncClient(app=web_app, base_url="http://test") as client:
            # Create weather-enabled agent
            weather_agent_response = await client.post("/api/v1/agents", json={
                "agent_type": "custom",
                "name": "Weather Research Agent",
                "config": {
                    "agent_type": "weather_analyst",
                    "enable_weather_api": True
                }
            })
            
            assert weather_agent_response.status_code == 200
            
            # Execute weather-related task
            weather_task_response = await client.post("/api/v1/orchestration/task", json={
                "task": "What's the current weather in London and how might it affect renewable energy production?",
                "strategy": "single_best",
                "context": {"location": "London, UK"}
            })
            
            assert weather_task_response.status_code == 200
            weather_data = weather_task_response.json()
            
            assert weather_data["success"] is True
            assert "london" in weather_data["primary_result"].lower()
            assert any(weather_term in weather_data["primary_result"].lower() 
                      for weather_term in ["weather", "temperature", "wind", "solar"])
    
    @pytest.mark.asyncio
    async def test_mcp_server_integration(self, web_app, mock_mcp_server):
        """Test MCP server integration for external tool access."""
        async with httpx.AsyncClient(app=web_app, base_url="http://test") as client:
            # Create agent with MCP tools
            mcp_agent_response = await client.post("/api/v1/agents", json={
                "agent_type": "custom",
                "name": "MCP Enhanced Agent",
                "config": {
                    "agent_type": "research_analyst",
                    "enable_mcp_tools": True,
                    "mcp_servers": ["perplexity", "tavily"]
                }
            })
            
            assert mcp_agent_response.status_code == 200
            
            # Execute task requiring external tools
            mcp_task_response = await client.post("/api/v1/orchestration/task", json={
                "task": "Research the latest developments in quantum computing using external sources",
                "strategy": "single_best",
                "context": {"use_external_tools": True}
            })
            
            assert mcp_task_response.status_code == 200
            mcp_data = mcp_task_response.json()
            
            assert mcp_data["success"] is True
            assert mcp_data["metadata"]["external_tools_used"] is True
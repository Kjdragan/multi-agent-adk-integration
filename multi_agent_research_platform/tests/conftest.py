# Test configuration and fixtures for Multi-Agent Research Platform
# This file provides common test fixtures and configuration used across all test suites

import pytest
import asyncio
import os
import tempfile
import sqlite3
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, AsyncMock, patch
from dataclasses import dataclass

# Set test environment
os.environ["ENVIRONMENT"] = "test"
os.environ["LOG_LEVEL"] = "DEBUG"

# Import platform components with correct paths
from src.config.app import AppConfig
from src.platform_logging import RunLogger
from src.services import SessionService, MemoryService, ArtifactService
from src.agents.factory import AgentFactory
from src.agents.orchestrator import AgentOrchestrator
from src.agents.base import Agent, AgentResult, AgentCapability, AgentRegistry


@dataclass
class TestConfig:
    """Test-specific configuration."""
    test_database_url: str = "sqlite:///:memory:"
    test_timeout: int = 30
    mock_api_calls: bool = True
    enable_logging: bool = True
    test_data_dir: str = "tests/fixtures/data"


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers", "unit: Unit tests for individual components"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests for component interactions"
    )
    config.addinivalue_line(
        "markers", "e2e: End-to-end tests for complete workflows"
    )
    config.addinivalue_line(
        "markers", "performance: Performance and load tests"
    )
    config.addinivalue_line(
        "markers", "slow: Slow-running tests"
    )
    config.addinivalue_line(
        "markers", "requires_api_key: Tests requiring real API keys"
    )
    config.addinivalue_line(
        "markers", "mock_apis: Tests with mocked external API calls"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)


# Async test event loop configuration
@pytest.fixture(scope="function")
def event_loop():
    """Create event loop for async tests."""
    # Create new event loop for each test to avoid state leakage
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        yield loop
    finally:
        # Proper cleanup of pending tasks
        try:
            # Cancel all pending tasks
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            
            # Wait for cancellation to complete
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        except Exception:
            pass  # Don't fail tests due to cleanup issues
        finally:
            loop.close()


# Configuration fixtures
@pytest.fixture(scope="session")
def test_config():
    """Provide test configuration."""
    return TestConfig()


@pytest.fixture(scope="session")
def app_config():
    """Provide application configuration for testing."""
    # Temporarily disable .env file loading for tests
    import os
    from tempfile import NamedTemporaryFile
    
    # Create empty temp env file
    with NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
        f.write("# Test environment file\n")
        temp_env_path = f.name
    
    try:
        # Create config without loading main .env
        config = AppConfig(_env_file=temp_env_path)
        
        # Set basic test configuration
        config.app_name = "test-app"
        config.development_mode = True
        config.mock_external_services = True
        
        return config
    finally:
        # Clean up temp file
        if os.path.exists(temp_env_path):
            os.unlink(temp_env_path)


# Service fixtures
@pytest.fixture(scope="function")
async def test_session_service():
    """Provide session service for testing."""
    from src.services.session import InMemorySessionService
    
    session_service = InMemorySessionService()
    
    try:
        await session_service.start()  # Proper initialization
        yield session_service
    finally:
        # Proper cleanup
        try:
            await session_service.stop()
        except Exception:
            pass  # Don't fail tests due to cleanup issues


@pytest.fixture(scope="function")
def temp_database_file():
    """Provide temporary database file for file-based tests."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    
    # Track open connections for cleanup
    connections = []
    
    try:
        yield db_path
    finally:
        # Close any open connections to the database
        try:
            # Close all SQLite connections to this file
            import gc
            for obj in gc.get_objects():
                if isinstance(obj, sqlite3.Connection):
                    try:
                        if hasattr(obj, 'execute'):
                            obj.close()
                    except Exception:
                        pass
        except Exception:
            pass
        
        # Remove the database file
        try:
            if os.path.exists(db_path):
                os.unlink(db_path)
        except PermissionError:
            # On Windows, file might still be locked
            import time
            time.sleep(0.1)
            try:
                if os.path.exists(db_path):
                    os.unlink(db_path)
            except Exception:
                pass  # Don't fail tests due to cleanup issues


@pytest.fixture(scope="function", autouse=True)
def reset_agent_registry():
    """Reset agent registry between tests to prevent interference."""
    # Clear the registry before each test
    try:
        AgentRegistry.clear_all()
    except Exception:
        pass  # Registry might not be initialized yet
    
    yield
    
    # Clean up after test
    try:
        # Get all agents and properly clean them up
        agents = AgentRegistry.get_all_agents()
        for agent in agents:
            try:
                if hasattr(agent, 'cleanup'):
                    agent.cleanup()
                if hasattr(agent, 'deactivate'):
                    agent.deactivate()
            except Exception:
                pass  # Don't fail tests due to cleanup issues
        
        # Clear the registry
        AgentRegistry.clear_all()
    except Exception:
        pass  # Don't fail tests due to cleanup issues


# Mock fixtures
@pytest.fixture(scope="function")
def mock_google_ai_client():
    """Mock Google AI client for testing."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.text = "This is a test response from the AI model."
    mock_response.finish_reason = "stop"
    mock_response.usage_metadata = Mock()
    mock_response.usage_metadata.total_token_count = 100
    
    mock_client.generate_content = AsyncMock(return_value=mock_response)
    
    # Try to mock Google Generative AI if available
    try:
        with patch('google.generativeai.GenerativeModel') as mock_model:
            mock_model.return_value = mock_client
            yield mock_client
    except (ImportError, AttributeError):
        # If module doesn't exist, just yield the mock client
        yield mock_client


@pytest.fixture(scope="function")
def mock_openweather_client():
    """Mock OpenWeather API client for testing."""
    mock_weather_data = {
        "weather": [{"main": "Clear", "description": "clear sky"}],
        "main": {
            "temp": 22.5,
            "feels_like": 23.1,
            "humidity": 65,
            "pressure": 1013
        },
        "wind": {"speed": 3.2, "deg": 180},
        "name": "London",
        "sys": {"country": "GB"}
    }
    
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.json.return_value = mock_weather_data
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        yield mock_get


@pytest.fixture(scope="function")
def mock_mcp_server():
    """Mock MCP server for testing."""
    mock_server = Mock()
    mock_server.call_tool = AsyncMock(return_value={
        "result": "Mock MCP tool result",
        "success": True,
        "metadata": {"source": "mock_mcp"}
    })
    
    with patch('src.tools.mcp_integration.MCPClient') as mock_client:
        mock_client.return_value = mock_server
        yield mock_server


# Agent fixtures
@pytest.fixture(scope="function")
def agent_factory(mock_google_ai_client):
    """Provide agent factory for testing."""
    # Create a mock logger for testing
    logger = Mock()
    logger.info = Mock()
    logger.error = Mock()
    logger.debug = Mock()
    logger.warning = Mock()
    
    factory = AgentFactory(logger=logger)
    yield factory


@pytest.fixture(scope="function")
def test_agent(agent_factory):
    """Provide a test LLM agent."""
    from multi_agent_research_platform.src.agents.llm_agent import LLMRole
    try:
        agent = agent_factory.create_llm_agent(
            role=LLMRole.RESEARCHER,
            name="Test Research Agent",
            auto_optimize_model=False
        )
        yield agent
    except Exception:
        # If agent creation fails, yield a mock agent
        mock_agent = Mock()
        mock_agent.agent_id = "test_agent"
        mock_agent.name = "Test Research Agent"
        mock_agent.capabilities = [AgentCapability.RESEARCH]
        yield mock_agent


@pytest.fixture(scope="function")
def agent_orchestrator():
    """Provide agent orchestrator for testing."""
    # Create a mock logger for testing
    logger = Mock()
    logger.info = Mock()
    logger.error = Mock()
    logger.debug = Mock()
    logger.warning = Mock()
    
    orchestrator = AgentOrchestrator(logger=logger)
    yield orchestrator


@pytest.fixture(scope="function")
def test_agent_registry():
    """Provide clean agent registry for testing."""
    # Clear existing registry
    AgentRegistry._agents.clear()
    AgentRegistry._agents_by_type.clear()
    AgentRegistry._agents_by_capability.clear()
    
    yield AgentRegistry
    
    # Cleanup after test
    AgentRegistry._agents.clear()
    AgentRegistry._agents_by_type.clear()
    AgentRegistry._agents_by_capability.clear()


# Data fixtures
@pytest.fixture(scope="session")
def sample_tasks():
    """Provide sample tasks for testing."""
    return [
        "What is the current weather in London?",
        "Research the benefits of renewable energy sources",
        "Analyze the stock market trends for tech companies",
        "Write a summary of artificial intelligence developments",
        "Compare different programming languages for web development"
    ]


@pytest.fixture(scope="session")
def sample_agent_configs():
    """Provide sample agent configurations."""
    return {
        "researcher": {
            "role": "researcher",
            "name": "Test Researcher",
            "capabilities": ["research", "analysis"],
            "temperature": 0.7,
            "max_tokens": 2000
        },
        "writer": {
            "role": "writer",
            "name": "Test Writer",
            "capabilities": ["writing", "editing"],
            "temperature": 0.8,
            "max_tokens": 3000
        },
        "analyst": {
            "role": "analyst",
            "name": "Test Analyst",
            "capabilities": ["analysis", "data_processing"],
            "temperature": 0.3,
            "max_tokens": 2500
        }
    }


@pytest.fixture(scope="function")
def sample_orchestration_context():
    """Provide sample orchestration context."""
    return {
        "user_id": "test_user_123",
        "session_id": "test_session_456",
        "deadline": "2024-12-31T23:59:59Z",
        "priority": "medium",
        "requirements": ["accuracy", "comprehensive"],
        "target_audience": "technical",
        "max_execution_time": 120
    }


# Test utilities
@pytest.fixture(scope="function")
def mock_time():
    """Mock time for consistent testing."""
    with patch('time.time') as mock_t:
        mock_t.return_value = 1640995200.0  # 2022-01-01 00:00:00 UTC
        yield mock_t


@pytest.fixture(scope="function")
def capture_logs():
    """Capture log messages for testing."""
    import logging
    from io import StringIO
    
    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    logger = RunLogger()
    
    yield log_capture


# Performance testing fixtures
@pytest.fixture(scope="function")
def performance_metrics():
    """Provide performance metrics collection."""
    metrics = {
        "execution_times": [],
        "memory_usage": [],
        "api_calls": [],
        "errors": []
    }
    
    def add_metric(metric_type: str, value: Any):
        if metric_type in metrics:
            metrics[metric_type].append(value)
    
    def get_average(metric_type: str) -> float:
        if metric_type in metrics and metrics[metric_type]:
            return sum(metrics[metric_type]) / len(metrics[metric_type])
        return 0.0
    
    metrics["add"] = add_metric
    metrics["average"] = get_average
    
    yield metrics


# Test data loading utilities
@pytest.fixture(scope="session")
def load_test_data():
    """Utility to load test data files."""
    def _load_data(filename: str) -> Dict[str, Any]:
        import json
        test_data_path = os.path.join("tests", "fixtures", "data", filename)
        if os.path.exists(test_data_path):
            with open(test_data_path, 'r') as f:
                return json.load(f)
        return {}
    
    return _load_data


# Async test helpers
async def wait_for_condition(condition_func, timeout: float = 5.0, interval: float = 0.1):
    """Wait for a condition to become true with timeout."""
    import time
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        if await condition_func():
            return True
        await asyncio.sleep(interval)
    
    return False


@pytest.fixture(scope="function")
def async_wait():
    """Provide async wait utility for tests."""
    return wait_for_condition


# Test environment cleanup
@pytest.fixture(autouse=True)
def cleanup_environment():
    """Clean up environment after each test."""
    yield
    
    # Reset environment variables
    test_env_vars = [
        "GOOGLE_API_KEY",
        "OPENWEATHER_API_KEY", 
        "PERPLEXITY_API_KEY",
        "TAVILY_API_KEY"
    ]
    
    for var in test_env_vars:
        if var in os.environ and os.environ[var].startswith("test_"):
            del os.environ[var]


# Parameterized fixtures for comprehensive testing
@pytest.fixture(params=[
    "single_best",
    "consensus", 
    "parallel_all",
    "adaptive"
])
def orchestration_strategy(request):
    """Parameterized orchestration strategies for testing."""
    return request.param


@pytest.fixture(params=[
    {"temperature": 0.3, "max_tokens": 1000},
    {"temperature": 0.7, "max_tokens": 2000},
    {"temperature": 0.9, "max_tokens": 3000}
])
def agent_config_params(request):
    """Parameterized agent configurations for testing."""
    return request.param


# Error simulation fixtures
@pytest.fixture(scope="function")
def simulate_api_errors():
    """Simulate various API errors for error handling tests."""
    def _simulate_error(error_type: str):
        if error_type == "timeout":
            raise asyncio.TimeoutError("API call timed out")
        elif error_type == "rate_limit":
            raise Exception("Rate limit exceeded")
        elif error_type == "auth_error":
            raise Exception("Authentication failed")
        elif error_type == "network_error":
            raise Exception("Network connection failed")
        else:
            raise Exception(f"Unknown error type: {error_type}")
    
    return _simulate_error


# Test markers for conditional execution
def requires_api_key(key_name: str):
    """Mark test as requiring a specific API key."""
    return pytest.mark.skipif(
        not os.getenv(key_name),
        reason=f"Test requires {key_name} environment variable"
    )


def requires_internet():
    """Mark test as requiring internet connection."""
    import socket
    
    def _check_internet():
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True
        except OSError:
            return False
    
    return pytest.mark.skipif(
        not _check_internet(),
        reason="Test requires internet connection"
    )


# Export commonly used fixtures and utilities
__all__ = [
    "test_config",
    "app_config", 
    "test_database",
    "mock_google_ai_client",
    "mock_openweather_client",
    "agent_factory",
    "test_agent",
    "agent_orchestrator",
    "sample_tasks",
    "sample_agent_configs",
    "performance_metrics",
    "wait_for_condition",
    "requires_api_key",
    "requires_internet"
]
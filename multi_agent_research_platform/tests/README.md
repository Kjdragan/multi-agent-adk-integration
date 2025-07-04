# Testing Framework

This comprehensive testing framework provides multiple levels of testing for the Multi-Agent Research Platform, ensuring code quality, reliability, and performance.

## ✅ Current Status (Post ADK v1.5.0 Migration)

- **84 tests discovered** across all test categories
- **Test collection working** properly after import fixes
- **Test issues resolved** - Gemini integration tests now passing (12/12)
- **Core functionality verified** - Task complexity analysis and orchestration working
- **Ready for development** with proper PYTHONPATH setup

**Quick Verification**:
```bash
# Verify test discovery works
PYTHONPATH=. python -m pytest tests/ --co -q
# Should show "84 tests collected"
```

## 📁 Test Structure

```
tests/
├── README.md                    # This file
├── conftest.py                  # Global test configuration and fixtures
├── unit/                        # Unit tests for individual components
│   └── test_agents.py          # Agent functionality tests
├── integration/                 # Integration tests for component interactions
│   └── test_agent_workflows.py # Multi-agent workflow tests
├── e2e/                        # End-to-end tests for complete workflows
│   └── test_complete_workflows.py # Full user journey tests
├── performance/                # Performance and load tests
│   └── test_load_and_scalability.py # Scalability and performance tests
├── fixtures/                   # Test fixtures and data
│   └── data/                   # Sample test data files
│       ├── sample_tasks.json   # Test tasks and scenarios
│       └── test_configurations.json # Test configurations
└── utils/                      # Test utilities and helpers
    └── test_helpers.py         # Common test utilities
```

## 🚀 Quick Start

### Prerequisites

1. **Install Dependencies**:
   ```bash
   # Install all dependencies with uv (recommended - all test deps already in pyproject.toml)
   uv sync

   # To add new test dependencies in the future, use:
   uv add <package-name>  # Adds to main dependencies
   # or
   uv add <package-name> --group dev  # Adds to dev group

   # Current test dependencies already included:
   # pytest>=8.4.1, pytest-asyncio>=1.0.0, pytest-cov>=6.2.1,
   # pytest-mock>=3.14.1, pytest-timeout>=2.4.0, psutil>=7.0.0

   # Alternative: Install manually with pip
   pip install pytest pytest-asyncio pytest-cov pytest-timeout pytest-mock psutil
   ```

2. **Configure Authentication** (required for tests that use Google AI):

   ⚠️ **IMPORTANT**: You must update your `.env` file for testing:

   ```bash
   # In your .env file, change this line for testing:
   GOOGLE_GENAI_USE_VERTEXAI=False  # REQUIRED FOR TESTING WITH API KEYS
   # GOOGLE_GENAI_USE_VERTEXAI=True  # Default production setting

   # Then set your API key:
   GOOGLE_API_KEY=your_google_api_key_from_gcp
   ```

   **Testing Setup Steps**:
   1. Get an API key from Google Cloud Console > APIs & Services > Credentials
   2. Edit `.env` and set `GOOGLE_GENAI_USE_VERTEXAI=False`
   3. Set `GOOGLE_API_KEY=your_api_key`
   4. Run tests
   5. **Remember to change back to `True` for production deployment**

   **Alternative environment variables**:
   ```bash
   # For local testing (override .env temporarily)
   export GOOGLE_API_KEY="your_google_api_key"
   export GOOGLE_GENAI_USE_VERTEXAI=False

   # External service APIs (for integration tests)
   export OPENWEATHER_API_KEY="your_openweather_key"
   export PERPLEXITY_API_KEY="your_perplexity_key"
   export TAVILY_API_KEY="your_tavily_key"
   ```

### Running Tests

**Using the Test Runner Script** (Recommended):

✅ **After v1.5.0 migration, the test runner automatically sets PYTHONPATH**

```bash
# MAKE SURE YOU'RE IN THE RIGHT DIRECTORY FIRST!
cd /home/kjdrag/lrepos/multi-agent-adk-integration/multi_agent_research_platform

# Option 1: Use test runner directly (handles environment automatically)
uv run python run_tests.py unit                    # Run all unit tests
uv run python run_tests.py integration -v          # Run integration tests with verbose output
uv run python run_tests.py all --skip-slow         # Run all tests (excluding slow performance tests)
uv run python run_tests.py specific tests/unit/test_agents.py  # Run specific test file
uv run python run_tests.py coverage                # Generate coverage report

# Option 2: If uv is not available, use python directly (but set PYTHONPATH)
PYTHONPATH=. python run_tests.py unit
PYTHONPATH=. python run_tests.py integration -v

# WRONG - These don't work:
# uv run_tests.py                    # Missing 'python'
# python run_tests.py                # Missing command argument
# uv run run_tests.py -v             # Missing command argument
```

**When to use `uv run`:**
- ✅ **Recommended**: If you installed dependencies with `uv sync`
- ✅ **Required**: If you have multiple Python environments
- ✅ **Best practice**: For consistent dependency management

**Using pytest directly**:
```bash
# Set PYTHONPATH and run all tests (required after ADK v1.5.0 migration)
PYTHONPATH=. python -m pytest tests/

# Run specific test types
PYTHONPATH=. python -m pytest tests/unit/ -m unit
PYTHONPATH=. python -m pytest tests/integration/ -m integration
PYTHONPATH=. python -m pytest tests/e2e/ -m e2e

# Run with coverage
PYTHONPATH=. python -m pytest --cov=src --cov-report=html

# Quick test to verify setup
PYTHONPATH=. python -m pytest tests/ --co -q  # Collect tests only
```

## 🧪 Test Types

### Unit Tests (`tests/unit/`)

Test individual components in isolation with mocked dependencies.

**Characteristics**:
- Fast execution (< 30 seconds total)
- No external dependencies
- Comprehensive mocking
- High code coverage

**Example**:
```python
@pytest.mark.asyncio
async def test_llm_agent_creation(self, mock_google_ai_client):
    config = LLMAgentConfig(role="researcher", name="Test Agent")
    agent = LLMAgent(config=config, tools=[])

    assert agent.name == "Test Agent"
    assert AgentCapability.RESEARCH in agent.capabilities
```

### Integration Tests (`tests/integration/`)

Test how multiple components work together.

**Characteristics**:
- Medium execution time (< 5 minutes)
- Real component interactions
- Selective mocking of external services
- Multi-agent workflows

**Example**:
```python
@pytest.mark.asyncio
async def test_agent_collaboration(self, agent_factory, test_agent_registry):
    # Create multiple agents
    researcher = await agent_factory.create_llm_agent(role="researcher")
    analyst = await agent_factory.create_llm_agent(role="analyst")

    # Test collaboration workflow
    result = await orchestrator.orchestrate_task(
        task="Complex research task",
        strategy=OrchestrationStrategy.CONSENSUS
    )

    assert result.success
    assert len(result.agents_used) >= 2
```

### End-to-End Tests (`tests/e2e/`)

Test complete user workflows from start to finish.

**Characteristics**:
- Longer execution time (< 15 minutes)
- Full system integration
- Real API calls (with API keys)
- User journey simulation

**Example**:
```python
@pytest.mark.asyncio
async def test_complete_research_workflow_via_api(self, web_app):
    async with httpx.AsyncClient(app=web_app) as client:
        # Create agent
        agent_response = await client.post("/api/v1/agents", json={...})

        # Execute task
        task_response = await client.post("/api/v1/orchestration/task", json={...})

        assert task_response.status_code == 200
        assert task_data["success"] is True
```

### Performance Tests (`tests/performance/`)

Test system performance, scalability, and resource usage.

**Characteristics**:
- Long execution time (up to 10 minutes)
- Load and stress testing
- Resource monitoring
- Performance benchmarking

**Example**:
```python
@pytest.mark.slow
@pytest.mark.asyncio
async def test_concurrent_agent_load(self, agent_factory, performance_metrics):
    # Test with increasing load levels
    for load in [10, 25, 50, 100]:
        tasks = [agent.execute_task(f"Task {i}") for i in range(load)]
        results = await asyncio.gather(*tasks)

        # Verify performance requirements
        assert success_rate >= 0.90
        assert avg_response_time < 1.0
```

## 🏷️ Test Markers

Tests are organized using pytest markers for flexible execution:

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.e2e` - End-to-end tests
- `@pytest.mark.performance` - Performance tests
- `@pytest.mark.slow` - Slow-running tests
- `@pytest.mark.requires_api_key` - Tests requiring real API keys
- `@pytest.mark.requires_internet` - Tests requiring internet access
- `@pytest.mark.mock_apis` - Tests with mocked external APIs

**Usage Examples**:
```bash
# Run only fast tests
pytest -m "not slow"

# Run tests that don't require API keys
pytest -m "not requires_api_key"

# Run specific test types
pytest -m "unit or integration"

# Run smoke tests
pytest -m smoke
```

## 🛠️ Test Configuration

### Global Configuration (`conftest.py`)

Provides common fixtures and configuration:

```python
@pytest.fixture
async def agent_factory(app_config, mock_google_ai_client):
    """Provide agent factory for testing."""
    return AgentFactory()

@pytest.fixture
def mock_google_ai_client():
    """Mock Google AI client for testing."""
    # Mock implementation
```

### Environment Configuration

Different test environments with specific configurations:

- **Unit Test Environment**: All external dependencies mocked
- **Integration Test Environment**: Real component interactions, selective mocking
- **Performance Test Environment**: Optimized for performance measurement
- **E2E Test Environment**: Full system integration

### Test Data

Structured test data in `tests/fixtures/data/`:

- **sample_tasks.json**: Various task types and complexity levels
- **test_configurations.json**: Agent configs, orchestration strategies, performance baselines

## 📊 Coverage and Quality

### Coverage Requirements

- **Minimum Coverage**: 80%
- **Target Coverage**: 90%+
- **Critical Components**: 95%+

### Coverage Reports

```bash
# Generate HTML coverage report
python run_tests.py coverage

# View coverage in terminal
pytest --cov=src --cov-report=term-missing

# Generate XML report for CI/CD
pytest --cov=src --cov-report=xml
```

### Quality Metrics

The testing framework tracks various quality metrics:

- **Test Success Rate**: > 95%
- **Performance Benchmarks**: Response times, throughput, resource usage
- **Code Quality**: Linting, type checking, formatting

## 🔧 Writing Tests

### Best Practices

1. **Test Naming**: Use descriptive names that explain what is being tested
   ```python
   def test_agent_handles_timeout_gracefully():
   def test_orchestration_with_invalid_strategy_raises_error():
   ```

2. **Test Structure**: Follow Arrange-Act-Assert pattern
   ```python
   async def test_agent_execution():
       # Arrange
       agent = await agent_factory.create_llm_agent(role="researcher")
       task = "Test task"

       # Act
       result = await agent.execute_task(task)

       # Assert
       assert result.success
       assert len(result.result) > 0
   ```

3. **Use Appropriate Fixtures**: Leverage existing fixtures for common setup
   ```python
   async def test_with_fixtures(self, agent_factory, test_agent_registry, sample_tasks):
       # Test implementation using fixtures
   ```

4. **Mock External Dependencies**: Use mocks for external services
   ```python
   @patch('src.tools.weather.requests.get')
   async def test_weather_api_call(self, mock_get):
       mock_get.return_value.json.return_value = {"temp": 20}
       # Test implementation
   ```

### Test Utilities

Use the provided test utilities in `tests/utils/test_helpers.py`:

```python
from tests.utils.test_helpers import MockAgent, TestDataGenerator, AsyncTestHelper

# Create mock agent
mock_agent = MockAgent(
    agent_id="test_agent",
    name="Test Agent",
    capabilities=[AgentCapability.RESEARCH],
    success_rate=0.9
)

# Generate test data
tasks = TestDataGenerator.generate_test_tasks(count=10)

# Wait for async condition
await AsyncTestHelper.wait_for_condition(
    condition=lambda: agent.is_ready(),
    timeout=5.0
)
```

### Creating New Tests

1. **Choose the Right Test Type**:
   - Unit: Testing individual functions/classes
   - Integration: Testing component interactions
   - E2E: Testing complete user workflows
   - Performance: Testing scalability and performance

2. **Add Appropriate Markers**:
   ```python
   @pytest.mark.asyncio
   @pytest.mark.integration
   async def test_multi_agent_workflow():
       # Test implementation
   ```

3. **Use Descriptive Assertions**:
   ```python
   assert result.success, f"Task execution failed: {result.error_message}"
   assert len(result.agents_used) >= 2, "Expected multiple agents to be used"
   ```

## 🚨 Troubleshooting

### Common Issues

1. **Authentication Errors** (Most Common):
   ```bash
   # Error: "No API key found" or "Vertex AI authentication failed"
   # Solution: Check your .env file
   GOOGLE_GENAI_USE_VERTEXAI=False  # MUST be False for testing
   GOOGLE_API_KEY=your_api_key_here
   ```

2. **Tests Timeout**: Increase timeout or optimize test
   ```bash
   pytest --timeout=300  # 5 minutes
   ```

3. **API Key Missing**: Set environment variables or skip API tests
   ```bash
   pytest -m "not requires_api_key"
   ```

4. **Import Errors**: Ensure PYTHONPATH includes project root (required after ADK v1.5.0 migration)
   ```bash
   export PYTHONPATH=.
   # Or run tests with: PYTHONPATH=. python -m pytest tests/
   ```

4. **Async Test Issues**: Ensure proper async/await usage and event loop handling

### Debug Mode

Run tests with debugging enabled:
```bash
# Verbose output with no capture
pytest -v -s --tb=long

# Drop into debugger on failure
pytest --pdb

# Run specific test with debugging
pytest tests/unit/test_agents.py::TestAgentFactory::test_create_llm_agent -v -s
```

### Performance Issues

If tests run slowly:
1. Check for unnecessary API calls
2. Optimize database operations
3. Use appropriate mocking
4. Run tests in parallel (when safe):
   ```bash
   pytest -n auto  # Requires pytest-xdist
   ```

## 🔄 Continuous Integration

### GitHub Actions Integration

The testing framework integrates with the CI/CD pipeline:

```yaml
- name: Run Tests
  run: |
    python run_tests.py unit
    python run_tests.py integration
    python run_tests.py lint
```

### Test Reports

Tests generate multiple report formats:
- **JUnit XML**: For CI/CD integration
- **HTML Coverage**: For detailed coverage analysis
- **Performance Reports**: For performance tracking

## 📈 Monitoring and Metrics

### Test Metrics Dashboard

Track testing metrics over time:
- Test execution time trends
- Coverage percentage trends
- Failure rate analysis
- Performance regression detection

### Quality Gates

Automated quality gates ensure:
- All tests pass before merge
- Coverage requirements met
- Performance benchmarks maintained
- No security vulnerabilities introduced

## 🤝 Contributing to Tests

### Adding New Tests

1. Identify the appropriate test type and location
2. Follow naming conventions and best practices
3. Add appropriate markers and documentation
4. Ensure tests are deterministic and fast
5. Update this README if adding new test patterns

### Test Review Checklist

- [ ] Tests cover happy path and edge cases
- [ ] Appropriate mocking of external dependencies
- [ ] Clear and descriptive test names
- [ ] Proper use of fixtures and utilities
- [ ] Performance considerations addressed
- [ ] Documentation updated if needed

---

For questions about testing or to report issues with the testing framework, please refer to the main project documentation or create an issue in the repository.

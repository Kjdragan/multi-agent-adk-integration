# Multi-Agent Research Platform

A comprehensive research platform built with Google Agent Development Kit (ADK) showcasing sophisticated multi-agent orchestration, context management, and enterprise-grade logging.

## Features

- **Advanced Multi-Agent System**: LLM, Workflow, and Custom agents with intelligent coordination
- **Sophisticated Context Management**: Proper use of InvocationContext, CallbackContext, ToolContext
- **Built-in Tool Integration**: Google Search, Code Execution, Vertex AI Search, BigQuery
- **MCP Server Integration**: Perplexity, Tavily, Brave Search, Omnisearch with authentication
- **Hybrid UI Strategy**: ADK Web Interface for debugging + Streamlit for production
- **Enterprise Logging**: Centralized, failure-safe logging with LLM-ready output
- **Session/State/Memory**: Advanced state management with database migration support

## Quick Start

### Prerequisites

- Python 3.13+
- UV package manager
- Google Cloud account (for Vertex AI features)
- API keys for external services (Perplexity, Tavily, etc.)

### Installation

1. Clone and navigate to the project:
```bash
cd multi-agent-research-platform
```

2. Copy environment template and configure:
```bash
cp .env.template .env
# Edit .env with your API keys and configuration
```

3. Install dependencies:
```bash
uv sync
```

4. Run the development server:
```bash
# ADK Web Interface (debugging)
uv run python main.py

# Streamlit Interface (production UX)
uv run streamlit run streamlit_app.py
```

## Architecture

### Project Structure
```
src/
├── agents/         # Agent implementations
├── tools/          # Custom tools and MCP integration  
├── config/         # Configuration management
├── services/       # Session, Memory, Artifact services
├── logging/        # Centralized logging system
└── ui/             # Streamlit interface components

logs/               # Per-run logging directory
docs/               # Comprehensive documentation
tests/              # Test suites
deployment/         # Cloud Run deployment configs
examples/           # Usage examples and demos
```

### Agent Ecosystem

- **Research Coordinator**: Master orchestrator with full session control
- **Tool Selection Agent**: Intelligent routing based on query analysis
- **Web Research Agent**: Multi-source research with MCP integration
- **Knowledge Base Agent**: Private data search via Vertex AI Search
- **Data Analysis Agent**: Statistical analysis with code execution
- **Document Processing Agent**: Multi-format document handling
- **Authentication Manager**: Secure credential management
- **Memory Integration Agent**: Cross-session knowledge continuity

## Documentation

Comprehensive documentation is available in the `docs/` directory:

1. [Architecture Overview](docs/01_Architecture_Overview.md)
2. [Agent Design Patterns](docs/02_Agent_Design_Patterns.md)
3. [Tool Integration Guide](docs/03_Tool_Integration_Guide.md)
4. [Session State Memory Management](docs/04_Session_State_Memory_Management.md)
5. [Logging System Reference](docs/05_Logging_System_Reference.md)
6. [UI Development Guide](docs/06_UI_Development_Guide.md)
7. [Deployment Operations](docs/07_Deployment_Operations.md)
8. [Configuration Reference](docs/08_Configuration_Reference.md)
9. [Testing Strategies](docs/09_Testing_Strategies.md)
10. [Troubleshooting Guide](docs/10_Troubleshooting_Guide.md)
11. [API Reference](docs/11_API_Reference.md)
12. [Best Practices](docs/12_Best_Practices.md)

## Usage Examples

### Basic Research Query
```python
# Via ADK Web Interface
# Navigate to http://localhost:8080
# Select "Research Coordinator" agent
# Enter your research question

# Via API
import asyncio
from src.agents.research_coordinator import ResearchCoordinator

async def research_example():
    coordinator = ResearchCoordinator()
    result = await coordinator.research("Latest developments in renewable energy")
    print(result)

asyncio.run(research_example())
```

### Custom Research Workflow
```python
from src.agents.workflows import ParallelResearchWorkflow

# Configure parallel research across multiple sources
workflow = ParallelResearchWorkflow(
    sources=["web", "knowledge_base", "data_analysis"],
    quality_threshold=0.8
)

result = await workflow.execute("Market analysis for electric vehicles")
```

## Configuration

The system uses Pydantic models for type-safe configuration:

```python
# .env configuration
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
GOOGLE_GENAI_USE_VERTEXAI=True
OPENWEATHER_API_KEY=your-api-key
PERPLEXITY_API_KEY=your-api-key
TAVILY_API_KEY=your-api-key
```

## Deployment

### Local Development
```bash
# Start both interfaces
uv run python main.py &      # ADK Web Interface on :8080
uv run streamlit run streamlit_app.py  # Streamlit on :8501
```

### Production (Google Cloud Run)
```bash
# Deploy using included script
./deployment/deploy_to_cloud_run.sh
```

## Logging

The system includes enterprise-grade logging with:

- **Per-run log directories**: Each invocation gets unique timestamped logs
- **Multiple log levels**: DEBUG, INFO, WARNING, ERROR with separate files
- **Failure-safe operation**: Logs persist even on crashes
- **LLM-ready format**: Structured output for easy analysis
- **Performance tracking**: Agent effectiveness and tool performance metrics

Log files are organized in `logs/runs/TIMESTAMP_INVOCATION-ID/` with:
- `debug.log`: Detailed debugging information
- `info.log`: General operation information  
- `error.log`: Errors and exceptions
- `events.jsonl`: Event stream in JSONL format
- `summary.json`: Run summary for LLM analysis

## Testing

Run the test suite:
```bash
# Unit tests
uv run pytest tests/unit/

# Integration tests  
uv run pytest tests/integration/

# All tests
uv run pytest
```

## Contributing

1. Follow the established code patterns in the documentation
2. Use proper context management (InvocationContext, CallbackContext, ToolContext)
3. Include comprehensive logging in all components
4. Add tests for new functionality
5. Update documentation for new features

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
1. Check the [Troubleshooting Guide](docs/10_Troubleshooting_Guide.md)
2. Review the comprehensive documentation in `docs/`
3. Examine log files in `logs/runs/` for debugging information
4. Open an issue with detailed logs and error information
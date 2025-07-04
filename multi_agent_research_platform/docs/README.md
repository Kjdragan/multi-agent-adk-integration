# Multi-Agent Research Platform

A comprehensive, production-ready platform for collaborative AI research using Google's Agent Development Kit (ADK). This platform enables multiple specialized AI agents to work together on complex research tasks, providing both technical debugging interfaces and user-friendly production interfaces.

## 🌟 Overview

The Multi-Agent Research Platform combines the power of Google's ADK with sophisticated orchestration capabilities to create a collaborative AI research environment. Whether you're a researcher, analyst, or decision-maker, this platform provides the tools you need to leverage AI agents for complex tasks.

### Key Features

- **🤖 Multi-Agent System**: Specialized agents for research, analysis, and content creation
- **🔧 Advanced Orchestration**: Multiple strategies for agent collaboration and task distribution
- **🌐 Dual Interfaces**: Technical debugging interface and user-friendly Streamlit interface
- **📊 Real-time Analytics**: Comprehensive monitoring and performance visualization
- **🔌 Extensible Architecture**: Easy integration with external tools and services
- **☁️ Cloud-Ready**: Optimized for Google Cloud Run deployment

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd multi-agent-research-platform

# Install with UV (recommended)
uv sync

# Or with pip
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy and configure environment variables
cp .env.example .env
# Edit .env with your API keys and settings
```

### 3. Run the Platform

**Option A: Streamlit Interface (Recommended for End Users)**
```bash
# Production interface
python src/streamlit/launcher.py

# Development mode
python src/streamlit/launcher.py -e development --reload
```

**Option B: Web Debug Interface (Recommended for Developers)**
```bash
# Debug interface with full monitoring
python src/web/launcher.py -e debug

# Production web interface
python src/web/launcher.py -e production
```

### 4. Access the Platform

- **Streamlit Interface**: http://localhost:8501
- **Web Debug Interface**: http://localhost:8081
- **API Documentation**: http://localhost:8081/docs

## 🏗️ Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interfaces                         │
├─────────────────────┬───────────────────────────────────────┤
│   Streamlit UI      │        Web Debug Interface           │
│  (Production UX)    │     (Development & Monitoring)       │
├─────────────────────┼───────────────────────────────────────┤
│                    REST APIs & WebSockets                  │
├─────────────────────────────────────────────────────────────┤
│                  Agent Orchestration Layer                 │
├─────────────────────┬───────────────┬───────────────────────┤
│    LLM Agents       │ Workflow      │    Custom Agents      │
│                     │ Agents        │                       │
├─────────────────────┼───────────────┼───────────────────────┤
│              Google ADK Integration                         │
├─────────────────────┬───────────────┬───────────────────────┤
│  Built-in Tools     │   Services    │   MCP Integration     │
│  (Search, Code,     │  (Session,    │  (Perplexity,         │
│   BigQuery, etc.)   │   Memory)     │   Tavily, etc.)       │
└─────────────────────┴───────────────┴───────────────────────┘
```

### Core Technologies

- **Google ADK**: Foundation for agent development and tool integration
- **FastAPI**: Web framework for APIs and debugging interface
- **Streamlit**: User-friendly production interface
- **SQLite**: Session and state management
- **WebSockets**: Real-time communication
- **Plotly**: Interactive charts and visualizations

## 🤖 Agent Types

### LLM Agents
Specialized language model agents with distinct roles:
- **Researcher**: Comprehensive research and data gathering
- **Analyst**: Data analysis and interpretation
- **Writer**: Content creation and documentation
- **Domain Expert**: Specialized knowledge in specific fields
- **And more...** (9 total specialized roles)

### Workflow Agents
Process orchestration agents for complex multi-step tasks:
- **Sequential Workflows**: Step-by-step task execution
- **Parallel Workflows**: Concurrent task processing
- **Conditional Workflows**: Dynamic decision-making

### Custom Agents
Domain-specific agents for specialized tasks:
- **Fact Checker**: Verification and validation
- **Data Analyst**: Statistical analysis and insights
- **Code Reviewer**: Technical code analysis
- **And more...** (12 total agent types)

## 🎯 Use Cases

### Research & Academia
- Literature reviews and meta-analyses
- Market research and competitive analysis
- Scientific paper summaries
- Grant proposal development

### Business & Strategy
- Market opportunity assessment
- Competitive intelligence
- Business plan development
- Risk analysis and mitigation

### Content & Communication
- Technical documentation creation
- Multi-language content adaptation
- Brand messaging consistency
- Content strategy development

### Technical & Engineering
- Architecture decision analysis
- Technology stack evaluation
- Code review and optimization
- System design validation

## 📊 Orchestration Strategies

The platform supports multiple orchestration strategies for different use cases:

- **🎯 Adaptive**: Dynamic strategy selection based on task characteristics
- **🤝 Consensus**: Multiple agents collaborate and build consensus
- **⚡ Parallel All**: All agents work simultaneously on the task
- **🏆 Single Best**: Best agent is selected for the task
- **🏁 Competitive**: Agents compete and best result is selected
- **🔄 Iterative**: Multi-round refinement process
- **🌊 Cascade**: Sequential agent chain with handoffs
- **🎲 Random**: Random agent selection for testing
- **⚖️ Weighted**: Probability-based agent selection

## 🔧 Configuration

### Environment Variables

```bash
# Core Configuration
GOOGLE_API_KEY=your_gemini_api_key
GOOGLE_CLOUD_PROJECT=your_project_id
GOOGLE_CLOUD_LOCATION=us-central1

# Development vs Production
GOOGLE_GENAI_USE_VERTEXAI=false  # false for local, true for cloud

# External Services
OPENWEATHER_API_KEY=your_weather_api_key
PERPLEXITY_API_KEY=your_perplexity_key
TAVILY_API_KEY=your_tavily_key

# Application Settings
PORT=8080
LOG_LEVEL=INFO
```

### Configuration Profiles

- **Development**: Debug logging, auto-reload, demo agents
- **Production**: Optimized performance, security, rate limiting
- **Debug**: Full monitoring, detailed logging, debugging tools
- **Demo**: Sample data, extended features, presentation mode

## 🌐 Interfaces

### Streamlit Interface (Production)
- **Target**: End users, researchers, business analysts
- **Features**: Intuitive UI, guided workflows, visual analytics
- **Access**: http://localhost:8501
- **Best for**: Day-to-day research tasks and analysis

### Web Debug Interface (Development)
- **Target**: Developers, system administrators, power users
- **Features**: Real-time monitoring, debugging tools, system health
- **Access**: http://localhost:8081
- **Best for**: Development, troubleshooting, system monitoring

## 🚀 Deployment

### Local Development
```bash
# Streamlit interface
python src/streamlit/launcher.py -e development

# Web debug interface
python src/web/launcher.py -e debug
```

### Production Deployment
```bash
# Google Cloud Run (recommended)
gcloud run deploy multitool-agent-service \
  --source . \
  --region us-central1 \
  --allow-unauthenticated

# Docker
docker build -t multi-agent-platform .
docker run -p 8080:8080 --env-file .env multi-agent-platform
```

## 📚 Documentation

- **[Installation Guide](INSTALLATION.md)** - Detailed setup instructions
- **[Quick Start Guide](QUICKSTART.md)** - Get up and running quickly
- **[Architecture Overview](ARCHITECTURE.md)** - System design and components
- **[Agent Documentation](AGENTS.md)** - Agent types and capabilities
- **[Web Interface Guide](WEB_INTERFACE.md)** - Debug interface documentation
- **[Streamlit Interface Guide](STREAMLIT_INTERFACE.md)** - Production UI guide
- **[API Reference](API_REFERENCE.md)** - Complete API documentation
- **[Configuration Guide](CONFIGURATION.md)** - Settings and environment setup
- **[Deployment Guide](DEPLOYMENT.md)** - Production deployment instructions
- **[Troubleshooting](TROUBLESHOOTING.md)** - Common issues and solutions
- **[Contributing](CONTRIBUTING.md)** - Developer contribution guide

## 🛠️ Development

### Project Structure
```
multi-agent-research-platform/
├── src/
│   ├── agents/          # Multi-agent system
│   ├── config/          # Configuration management
│   ├── logging/         # Centralized logging
│   ├── services/        # Core services (Session, Memory, etc.)
│   ├── web/             # Web debug interface
│   └── streamlit/       # Production Streamlit interface
├── docs/                # Documentation
├── tests/               # Test suite
├── scripts/             # Utility scripts
└── deployment/          # Deployment configurations
```

### Key Development Commands
```bash
# Install dependencies
uv sync

# Run tests
pytest tests/

# Run linting
ruff check src/

# Run type checking
mypy src/

# Start development server
python src/web/launcher.py -e development --reload
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on:

- Setting up the development environment
- Code style and standards
- Testing requirements
- Pull request process
- Issue reporting

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

## 🙏 Acknowledgments

- **Google ADK Team** - For the excellent Agent Development Kit
- **Streamlit Team** - For the intuitive web framework
- **FastAPI Team** - For the high-performance web framework
- **Open Source Community** - For the many libraries that make this possible

## 📞 Support

- **Documentation**: Check our [comprehensive docs](docs/)
- **Issues**: Report bugs and request features via GitHub Issues
- **Discussions**: Join community discussions for questions and ideas
- **Email**: Contact the maintainers for urgent issues

---

**Built with ❤️ using Google ADK and the power of collaborative AI**
"""
Multi-Agent Research Platform Web Interface

Comprehensive web interface module providing debugging, monitoring, and
interaction capabilities for the multi-agent research platform using
Google's Agent Development Kit (ADK).

Main Components:
- WebInterface: Core web interface with ADK FastAPI integration
- MultiAgentWebApp: Complete web application with dashboard and debugging
- Dashboard components: Real-time monitoring and visualization
- API endpoints: REST APIs for agent management and orchestration
- WebSocket handlers: Real-time communication and event broadcasting
- Template rendering: Jinja2-based HTML template system

Quick Start:
```python
from src.web import MultiAgentWebApp

# Create and run web application
app = MultiAgentWebApp(environment="debug")
await app.run()
```

Web Launcher:
```bash
# Run from project root
python src/web/launcher.py --environment debug --port 8081
```
"""

from .app import MultiAgentWebApp
from .interface import WebInterface, DebugInterface, MonitoringInterface
from .config import (
    WebConfig, DebugConfig, MonitoringConfig, WebSocketConfig, APIConfig,
    WebInterfaceMode, LogLevel, WebConfigFactory, get_config_for_environment
)
from .api import AgentAPI, OrchestrationAPI, DebugAPI, MonitoringAPI
from .handlers import WebSocketHandler, EventHandler, LogHandler
from .dashboards import (
    AgentDashboard, TaskDashboard, PerformanceDashboard,
    BaseDashboard, DashboardType, DashboardWidget
)
from .templates import TemplateRenderer, DashboardTemplateRenderer, get_template_renderer, get_dashboard_renderer

# Version information
__version__ = "1.0.0"
__author__ = "Multi-Agent Research Platform Team"

# Export main classes and functions
__all__ = [
    # Main application
    "MultiAgentWebApp",
    
    # Core interfaces
    "WebInterface",
    "DebugInterface", 
    "MonitoringInterface",
    
    # Configuration
    "WebConfig",
    "DebugConfig",
    "MonitoringConfig", 
    "WebSocketConfig",
    "APIConfig",
    "WebInterfaceMode",
    "LogLevel",
    "WebConfigFactory",
    "get_config_for_environment",
    
    # API handlers
    "AgentAPI",
    "OrchestrationAPI",
    "DebugAPI",
    "MonitoringAPI",
    
    # Communication handlers
    "WebSocketHandler",
    "EventHandler",
    "LogHandler",
    
    # Dashboard components
    "AgentDashboard",
    "TaskDashboard", 
    "PerformanceDashboard",
    "BaseDashboard",
    "DashboardType",
    "DashboardWidget",
    
    # Template rendering
    "TemplateRenderer",
    "DashboardTemplateRenderer",
    "get_template_renderer",
    "get_dashboard_renderer",
]
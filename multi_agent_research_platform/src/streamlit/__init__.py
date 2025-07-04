"""
Multi-Agent Research Platform - Streamlit Interface

Production-ready Streamlit interface for the multi-agent research platform.
Provides a user-friendly web interface for research tasks, agent management,
and result visualization optimized for end-user interactions.

Key Features:
- Intuitive user interface for non-technical users
- Interactive agent creation and management
- Research task execution with real-time progress
- Visual analytics and performance charts
- Task history and result management
- Export capabilities for results and data
- Responsive design with modern UI components

Quick Start:
```python
# Run the Streamlit app
python src/streamlit/launcher.py --environment production

# Or with custom settings
python src/streamlit/launcher.py -e demo -p 8502 --theme light
```

Components:
- StreamlitApp: Main application class with full interface
- StreamlitConfig: Configuration management for different environments
- UI Components: Reusable widgets for agents, tasks, and charts
- Launcher: Easy startup script with environment selection
"""

from .app import StreamlitApp
from .config import (
    StreamlitConfig, UIConfig, SecurityConfig,
    StreamlitTheme, PageLayout,
    StreamlitConfigFactory, get_streamlit_config,
    DEFAULT_CONFIG, DEVELOPMENT_CONFIG, DEMO_CONFIG
)
from .components import (
    AgentCard, TaskForm, PerformanceChart, SystemMetrics, 
    TaskResultDisplay, ProgressIndicator,
    format_timestamp, get_status_color, create_download_link
)

# Version information
__version__ = "1.0.0"
__author__ = "Multi-Agent Research Platform Team"

# Export main classes and functions
__all__ = [
    # Main application
    "StreamlitApp",
    
    # Configuration
    "StreamlitConfig",
    "UIConfig", 
    "SecurityConfig",
    "StreamlitTheme",
    "PageLayout",
    "StreamlitConfigFactory",
    "get_streamlit_config",
    "DEFAULT_CONFIG",
    "DEVELOPMENT_CONFIG", 
    "DEMO_CONFIG",
    
    # Components
    "AgentCard",
    "TaskForm",
    "PerformanceChart", 
    "SystemMetrics",
    "TaskResultDisplay",
    "ProgressIndicator",
    
    # Utilities
    "format_timestamp",
    "get_status_color", 
    "create_download_link",
]

def create_streamlit_app(environment: str = "production", **config_overrides) -> StreamlitApp:
    """
    Create a Streamlit application instance with the specified environment.
    
    Args:
        environment: Environment configuration ("development", "production", "demo", "minimal")
        **config_overrides: Additional configuration overrides
        
    Returns:
        Configured StreamlitApp instance
    """
    app = StreamlitApp()
    
    # Load environment configuration
    config = get_streamlit_config(environment)
    
    # Apply configuration overrides
    for key, value in config_overrides.items():
        if "." in key:
            # Handle nested config like "streamlit.theme" or "ui.primary_color"
            section, config_key = key.split(".", 1)
            if section in config:
                setattr(config[section], config_key, value)
    
    return app

def run_streamlit_interface(
    environment: str = "production",
    host: str = "localhost", 
    port: int = 8501,
    auto_open: bool = True
):
    """
    Convenience function to run the Streamlit interface.
    
    Args:
        environment: Environment configuration
        host: Host to bind to
        port: Port to bind to
        auto_open: Whether to automatically open browser
    """
    import subprocess
    import sys
    from pathlib import Path
    
    # Get launcher script path
    launcher_path = Path(__file__).parent / "launcher.py"
    
    # Build command
    cmd = [
        sys.executable, str(launcher_path),
        "--environment", environment,
        "--host", host,
        "--port", str(port)
    ]
    
    if not auto_open:
        cmd.append("--no-browser")
    
    # Run launcher
    subprocess.run(cmd)

# Module-level documentation
def get_module_info():
    """Get module information and capabilities."""
    return {
        "name": "Multi-Agent Research Platform Streamlit Interface",
        "version": __version__,
        "description": "Production-ready Streamlit interface for multi-agent research platform",
        "features": [
            "User-friendly web interface for research tasks",
            "Interactive agent creation and management",
            "Real-time task execution with progress tracking",
            "Visual analytics with charts and metrics",
            "Task history and result management",
            "Export capabilities for data and results",
            "Responsive design optimized for all devices",
            "Multiple environment configurations",
            "Security features and rate limiting",
            "Caching and performance optimizations"
        ],
        "components": {
            "streamlit_app": "Main application with full interface",
            "configuration": "Environment-specific configuration management",
            "ui_components": "Reusable widgets for agents, tasks, charts",
            "launcher": "Easy startup script with environment selection"
        },
        "supported_environments": ["development", "production", "demo", "minimal"],
        "target_users": [
            "Researchers and analysts",
            "Business users and decision makers", 
            "Students and educators",
            "Non-technical users requiring AI assistance"
        ]
    }

# Usage examples
USAGE_EXAMPLES = {
    "basic": """
# Basic usage - run production interface
python src/streamlit/launcher.py

# Access at: http://localhost:8501
""",
    
    "development": """
# Development mode with auto-reload
python src/streamlit/launcher.py -e development --reload

# Features: Debug logging, demo agents, development tools
""",
    
    "demo": """
# Demo mode for presentations
python src/streamlit/launcher.py -e demo -p 8502

# Features: Sample data, extended analytics, pre-configured teams
""",
    
    "custom": """
# Custom configuration
python src/streamlit/launcher.py \\
    --environment production \\
    --host 0.0.0.0 \\
    --port 8080 \\
    --theme dark \\
    --no-browser
""",
    
    "programmatic": """
# Programmatic usage
from src.streamlit import create_streamlit_app, run_streamlit_interface

# Create app instance
app = create_streamlit_app(environment="demo")

# Or run directly
run_streamlit_interface(
    environment="production",
    host="0.0.0.0", 
    port=8501,
    auto_open=True
)
"""
}
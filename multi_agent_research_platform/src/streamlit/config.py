"""
Streamlit Configuration

Configuration settings and utilities for the Streamlit production interface
of the multi-agent research platform.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum

from ..config.base import BaseConfig


class StreamlitTheme(str, Enum):
    """Available Streamlit themes."""
    LIGHT = "light"
    DARK = "dark"
    AUTO = "auto"


class PageLayout(str, Enum):
    """Page layout options."""
    WIDE = "wide"
    CENTERED = "centered"


@dataclass
class StreamlitConfig(BaseConfig):
    """Main Streamlit application configuration."""
    
    # Application settings
    page_title: str = "Multi-Agent Research Platform"
    page_icon: str = "🤖"
    layout: PageLayout = PageLayout.WIDE
    initial_sidebar_state: str = "expanded"
    
    # Theme and styling
    theme: StreamlitTheme = StreamlitTheme.LIGHT
    custom_css_enabled: bool = True
    show_header: bool = True
    show_footer: bool = True
    
    # Feature flags
    enable_agent_creation: bool = True
    enable_task_execution: bool = True
    enable_analytics: bool = True
    enable_export: bool = True
    enable_real_time_updates: bool = False  # Disabled for production
    
    # Task execution settings
    default_timeout_seconds: int = 120
    max_timeout_seconds: int = 300
    enable_context_history: bool = True
    max_history_length: int = 10
    
    # Agent settings
    auto_create_demo_agents: bool = False
    max_agents_per_session: int = 20
    show_agent_details: bool = True
    
    # Analytics settings
    enable_charts: bool = True
    chart_library: str = "plotly"  # plotly, matplotlib, altair
    default_chart_theme: str = "plotly_white"
    
    # Performance settings
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300  # 5 minutes
    session_state_cleanup: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "page_title": self.page_title,
            "page_icon": self.page_icon,
            "layout": self.layout.value,
            "initial_sidebar_state": self.initial_sidebar_state,
            "theme": self.theme.value,
            "custom_css_enabled": self.custom_css_enabled,
            "show_header": self.show_header,
            "show_footer": self.show_footer,
            "enable_agent_creation": self.enable_agent_creation,
            "enable_task_execution": self.enable_task_execution,
            "enable_analytics": self.enable_analytics,
            "enable_export": self.enable_export,
            "enable_real_time_updates": self.enable_real_time_updates,
            "default_timeout_seconds": self.default_timeout_seconds,
            "max_timeout_seconds": self.max_timeout_seconds,
            "enable_context_history": self.enable_context_history,
            "max_history_length": self.max_history_length,
            "auto_create_demo_agents": self.auto_create_demo_agents,
            "max_agents_per_session": self.max_agents_per_session,
            "show_agent_details": self.show_agent_details,
            "enable_charts": self.enable_charts,
            "chart_library": self.chart_library,
            "default_chart_theme": self.default_chart_theme,
            "cache_enabled": self.cache_enabled,
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "session_state_cleanup": self.session_state_cleanup,
        }


@dataclass 
class UIConfig(BaseConfig):
    """User interface configuration."""
    
    # Layout settings
    sidebar_width: int = 350
    main_content_padding: str = "2rem"
    header_height: str = "80px"
    
    # Color scheme
    primary_color: str = "#667eea"
    secondary_color: str = "#764ba2"
    success_color: str = "#28a745"
    error_color: str = "#dc3545"
    warning_color: str = "#ffc107"
    info_color: str = "#17a2b8"
    
    # Typography
    font_family: str = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
    header_font_size: str = "2.5rem"
    body_font_size: str = "1rem"
    
    # Component settings
    card_border_radius: str = "8px"
    button_border_radius: str = "6px"
    input_border_radius: str = "4px"
    
    # Animation settings
    enable_animations: bool = True
    animation_duration: str = "0.3s"
    
    def get_css(self) -> str:
        """Generate CSS based on configuration."""
        return f"""
        <style>
            .main-header {{
                padding: 1rem 0;
                border-bottom: 2px solid #f0f2f6;
                margin-bottom: 2rem;
                height: {self.header_height};
            }}
            
            .metric-card {{
                background: linear-gradient(135deg, {self.primary_color} 0%, {self.secondary_color} 100%);
                padding: 1rem;
                border-radius: {self.card_border_radius};
                color: white;
                margin-bottom: 1rem;
            }}
            
            .agent-card {{
                border: 1px solid #e1e5e9;
                border-radius: {self.card_border_radius};
                padding: 1rem;
                margin-bottom: 1rem;
                background: white;
            }}
            
            .status-active {{ color: {self.success_color}; font-weight: bold; }}
            .status-inactive {{ color: #6c757d; font-weight: bold; }}
            .status-running {{ color: {self.info_color}; font-weight: bold; }}
            .status-completed {{ color: {self.success_color}; font-weight: bold; }}
            .status-error {{ color: {self.error_color}; font-weight: bold; }}
            
            .task-result {{
                background-color: #f8f9fa;
                border-left: 4px solid {self.info_color};
                padding: 1rem;
                margin: 1rem 0;
                border-radius: 0 {self.card_border_radius} {self.card_border_radius} 0;
            }}
            
            .sidebar-section {{
                margin-bottom: 2rem;
                padding: 1rem;
                background-color: #f8f9fa;
                border-radius: {self.card_border_radius};
            }}
            
            body {{
                font-family: {self.font_family};
                font-size: {self.body_font_size};
            }}
            
            h1, h2, h3 {{
                font-family: {self.font_family};
            }}
            
            h1 {{
                font-size: {self.header_font_size};
            }}
            
            .stButton > button {{
                border-radius: {self.button_border_radius};
                transition: all {self.animation_duration} ease;
            }}
            
            .stTextInput > div > div > input {{
                border-radius: {self.input_border_radius};
            }}
            
            .stSelectbox > div > div > div {{
                border-radius: {self.input_border_radius};
            }}
        </style>
        """


@dataclass
class SecurityConfig(BaseConfig):
    """Security configuration for Streamlit app."""
    
    # Session management
    session_timeout_minutes: int = 60
    max_concurrent_sessions: int = 100
    
    # Input validation
    max_task_length: int = 5000
    max_agent_name_length: int = 100
    sanitize_inputs: bool = True
    
    # Rate limiting
    enable_rate_limiting: bool = True
    max_requests_per_minute: int = 60
    max_agents_per_hour: int = 10
    max_tasks_per_hour: int = 30
    
    # Data protection
    log_user_actions: bool = False  # Disabled for privacy
    encrypt_session_data: bool = False
    store_task_history: bool = True
    max_stored_tasks: int = 1000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "session_timeout_minutes": self.session_timeout_minutes,
            "max_concurrent_sessions": self.max_concurrent_sessions,
            "max_task_length": self.max_task_length,
            "max_agent_name_length": self.max_agent_name_length,
            "sanitize_inputs": self.sanitize_inputs,
            "enable_rate_limiting": self.enable_rate_limiting,
            "max_requests_per_minute": self.max_requests_per_minute,
            "max_agents_per_hour": self.max_agents_per_hour,
            "max_tasks_per_hour": self.max_tasks_per_hour,
            "log_user_actions": self.log_user_actions,
            "encrypt_session_data": self.encrypt_session_data,
            "store_task_history": self.store_task_history,
            "max_stored_tasks": self.max_stored_tasks,
        }


class StreamlitConfigFactory:
    """Factory for creating Streamlit configuration instances."""
    
    @classmethod
    def create_development_config(cls) -> Dict[str, Any]:
        """Create configuration for development environment."""
        return {
            "streamlit": StreamlitConfig(
                theme=StreamlitTheme.LIGHT,
                enable_real_time_updates=True,
                auto_create_demo_agents=True,
                cache_enabled=False,  # Disable caching for development
            ),
            "ui": UIConfig(
                enable_animations=True,
            ),
            "security": SecurityConfig(
                enable_rate_limiting=False,  # Disabled for development
                log_user_actions=True,
                max_requests_per_minute=1000,
            ),
        }
    
    @classmethod
    def create_production_config(cls) -> Dict[str, Any]:
        """Create configuration for production environment."""
        return {
            "streamlit": StreamlitConfig(
                theme=StreamlitTheme.LIGHT,
                enable_real_time_updates=False,
                auto_create_demo_agents=False,
                cache_enabled=True,
                session_state_cleanup=True,
            ),
            "ui": UIConfig(
                enable_animations=True,
            ),
            "security": SecurityConfig(
                enable_rate_limiting=True,
                log_user_actions=False,  # Privacy-focused
                session_timeout_minutes=30,
                max_requests_per_minute=30,
                max_agents_per_hour=5,
                max_tasks_per_hour=15,
            ),
        }
    
    @classmethod
    def create_demo_config(cls) -> Dict[str, Any]:
        """Create configuration for demonstration purposes."""
        return {
            "streamlit": StreamlitConfig(
                theme=StreamlitTheme.LIGHT,
                auto_create_demo_agents=True,
                enable_analytics=True,
                enable_export=True,
                max_history_length=20,
            ),
            "ui": UIConfig(
                enable_animations=True,
                primary_color="#4facfe",
                secondary_color="#00f2fe",
            ),
            "security": SecurityConfig(
                enable_rate_limiting=False,
                session_timeout_minutes=120,  # Longer for demos
                max_requests_per_minute=200,
            ),
        }
    
    @classmethod
    def create_minimal_config(cls) -> Dict[str, Any]:
        """Create minimal configuration for basic usage."""
        return {
            "streamlit": StreamlitConfig(
                layout=PageLayout.CENTERED,
                enable_analytics=False,
                enable_export=False,
                cache_enabled=False,
                show_agent_details=False,
            ),
            "ui": UIConfig(
                enable_animations=False,
            ),
            "security": SecurityConfig(
                enable_rate_limiting=True,
                max_agents_per_hour=3,
                max_tasks_per_hour=10,
            ),
        }


def get_streamlit_config(environment: str = "production") -> Dict[str, Any]:
    """
    Get Streamlit configuration for specific environment.
    
    Args:
        environment: Environment name (development, production, demo, minimal)
        
    Returns:
        Configuration dictionary
    """
    if environment == "development":
        return StreamlitConfigFactory.create_development_config()
    elif environment == "production":
        return StreamlitConfigFactory.create_production_config()
    elif environment == "demo":
        return StreamlitConfigFactory.create_demo_config()
    elif environment == "minimal":
        return StreamlitConfigFactory.create_minimal_config()
    else:
        raise ValueError(f"Unknown environment: {environment}")


# Default configurations for quick access
DEFAULT_CONFIG = get_streamlit_config("production")
DEVELOPMENT_CONFIG = get_streamlit_config("development")
DEMO_CONFIG = get_streamlit_config("demo")
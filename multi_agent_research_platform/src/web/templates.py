"""
Template Rendering for Web Interface

Jinja2 template rendering support for the multi-agent research platform
web interface with dashboard and debugging capabilities.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from jinja2 import Environment, FileSystemLoader, Template

from ..platform_logging import RunLogger


class TemplateRenderer:
    """Template renderer using Jinja2 for the web interface."""
    
    def __init__(self, templates_dir: Optional[str] = None, logger: Optional[RunLogger] = None):
        self.logger = logger
        
        # Set templates directory
        if templates_dir:
            self.templates_dir = Path(templates_dir)
        else:
            # Default to templates directory relative to this file
            self.templates_dir = Path(__file__).parent / "templates"
        
        # Ensure templates directory exists
        self.templates_dir.mkdir(exist_ok=True)
        
        # Initialize Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            autoescape=True,
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Add custom filters and functions
        self._setup_template_filters()
        
        if self.logger:
            self.logger.debug(f"Template renderer initialized with directory: {self.templates_dir}")
    
    def _setup_template_filters(self):
        """Setup custom Jinja2 filters and functions."""
        
        def datetime_filter(timestamp):
            """Format timestamp as datetime string."""
            import datetime
            if isinstance(timestamp, (int, float)):
                dt = datetime.datetime.fromtimestamp(timestamp)
                return dt.strftime('%Y-%m-%d %H:%M:%S')
            return str(timestamp)
        
        def duration_filter(seconds):
            """Format seconds as human-readable duration."""
            if not isinstance(seconds, (int, float)):
                return "N/A"
            
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            
            if hours > 0:
                return f"{hours}h {minutes}m {secs}s"
            elif minutes > 0:
                return f"{minutes}m {secs}s"
            else:
                return f"{secs}s"
        
        def filesize_filter(bytes_size):
            """Format bytes as human-readable file size."""
            if not isinstance(bytes_size, (int, float)):
                return "N/A"
            
            for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
                if bytes_size < 1024.0:
                    return f"{bytes_size:.1f} {unit}"
                bytes_size /= 1024.0
            return f"{bytes_size:.1f} PB"
        
        def percentage_filter(value, decimals=1):
            """Format value as percentage."""
            if not isinstance(value, (int, float)):
                return "N/A"
            return f"{value:.{decimals}f}%"
        
        def get_status_class(status):
            """Get CSS class for status badge."""
            status_map = {
                'active': 'status-active',
                'inactive': 'status-inactive',
                'running': 'status-running',
                'completed': 'status-completed',
                'failed': 'status-error',
                'error': 'status-error',
                'success': 'status-completed',
                'pending': 'status-inactive',
            }
            return status_map.get(str(status).lower(), 'status-inactive')
        
        def get_agent_type_icon(agent_type):
            """Get Font Awesome icon for agent type."""
            icon_map = {
                'llm': 'fas fa-brain',
                'workflow': 'fas fa-sitemap',
                'custom': 'fas fa-cog',
                'research': 'fas fa-search',
                'analysis': 'fas fa-chart-bar',
                'domain_expert': 'fas fa-user-graduate',
                'fact_checker': 'fas fa-check-circle',
                'data_analyst': 'fas fa-database',
            }
            return icon_map.get(str(agent_type).lower(), 'fas fa-robot')
        
        def get_capability_icon(capability):
            """Get Font Awesome icon for capability."""
            icon_map = {
                'research': 'fas fa-search',
                'analysis': 'fas fa-chart-line',
                'data_processing': 'fas fa-database',
                'web_search': 'fas fa-globe',
                'code_execution': 'fas fa-code',
                'text_generation': 'fas fa-pen',
                'translation': 'fas fa-language',
                'summarization': 'fas fa-compress-alt',
                'question_answering': 'fas fa-question-circle',
                'classification': 'fas fa-tags',
                'sentiment_analysis': 'fas fa-smile',
                'entity_extraction': 'fas fa-extract',
            }
            return icon_map.get(str(capability).lower(), 'fas fa-star')
        
        # Register filters and globals with the environment
        self.env.filters['datetime'] = datetime_filter
        self.env.filters['duration'] = duration_filter
        self.env.filters['filesize'] = filesize_filter
        self.env.filters['percentage'] = percentage_filter
        self.env.globals['get_status_class'] = get_status_class
        self.env.globals['get_agent_type_icon'] = get_agent_type_icon
        self.env.globals['get_capability_icon'] = get_capability_icon
    
    def render_template(self, template_name: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Render a template with the given context.
        
        Args:
            template_name: Name of the template file
            context: Template context variables
            
        Returns:
            Rendered HTML string
        """
        try:
            template = self.env.get_template(template_name)
            context = context or {}
            
            # Add common context variables
            context.update(self._get_common_context())
            
            rendered = template.render(**context)
            
            if self.logger:
                self.logger.debug(f"Rendered template: {template_name}")
            
            return rendered
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to render template {template_name}: {e}")
            return self._render_error_template(template_name, str(e))
    
    def render_string(self, template_string: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Render a template string with the given context.
        
        Args:
            template_string: Template string to render
            context: Template context variables
            
        Returns:
            Rendered HTML string
        """
        try:
            template = self.env.from_string(template_string)
            context = context or {}
            
            # Add common context variables
            context.update(self._get_common_context())
            
            return template.render(**context)
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to render template string: {e}")
            return f"<div class='alert alert-danger'>Template Error: {e}</div>"
    
    def _get_common_context(self) -> Dict[str, Any]:
        """Get common context variables available to all templates."""
        import time
        
        return {
            'current_time': time.time(),
            'platform_name': 'Multi-Agent Research Platform',
            'platform_version': '1.0.0',
            'debug_mode': True,  # Would be configurable
        }
    
    def _render_error_template(self, template_name: str, error_message: str) -> str:
        """Render a fallback error template."""
        error_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Template Error - {{ platform_name }}</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .error { background: #f8d7da; border: 1px solid #f5c6cb; padding: 20px; border-radius: 5px; }
                .error h1 { color: #721c24; margin-top: 0; }
                .error p { color: #721c24; }
                pre { background: #f8f9fa; padding: 15px; border-radius: 3px; overflow-x: auto; }
            </style>
        </head>
        <body>
            <div class="error">
                <h1>Template Rendering Error</h1>
                <p><strong>Template:</strong> {{ template_name }}</p>
                <p><strong>Error:</strong></p>
                <pre>{{ error_message }}</pre>
                <p>Please check the template file and try again.</p>
            </div>
        </body>
        </html>
        """
        
        try:
            template = self.env.from_string(error_template)
            return template.render(
                template_name=template_name,
                error_message=error_message,
                platform_name="Multi-Agent Research Platform"
            )
        except Exception:
            # Ultimate fallback
            return f"""
            <html>
            <head><title>Template Error</title></head>
            <body>
                <h1>Template Error</h1>
                <p>Failed to render template: {template_name}</p>
                <p>Error: {error_message}</p>
            </body>
            </html>
            """
    
    def get_available_templates(self) -> list:
        """Get list of available template files."""
        templates = []
        
        try:
            for file_path in self.templates_dir.rglob("*.html"):
                relative_path = file_path.relative_to(self.templates_dir)
                templates.append(str(relative_path))
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to list templates: {e}")
        
        return sorted(templates)
    
    def template_exists(self, template_name: str) -> bool:
        """Check if a template file exists."""
        template_path = self.templates_dir / template_name
        return template_path.exists()


class DashboardTemplateRenderer(TemplateRenderer):
    """Specialized template renderer for dashboard components."""
    
    def render_dashboard_widget(self, widget_type: str, widget_data: Dict[str, Any]) -> str:
        """
        Render a specific dashboard widget.
        
        Args:
            widget_type: Type of widget (e.g., 'stats_card', 'chart', 'table')
            widget_data: Widget data and configuration
            
        Returns:
            Rendered widget HTML
        """
        template_name = f"widgets/{widget_type}.html"
        
        # Fallback to generic widget template if specific one doesn't exist
        if not self.template_exists(template_name):
            template_name = "widgets/generic.html"
        
        if not self.template_exists(template_name):
            # Create a basic widget template inline
            widget_template = """
            <div class="widget widget-{{ widget_type }}" id="{{ widget_id }}">
                <div class="widget-header">
                    <h6>{{ title }}</h6>
                </div>
                <div class="widget-body">
                    {% if data %}
                        <pre>{{ data | tojson(indent=2) }}</pre>
                    {% else %}
                        <p class="text-muted">No data available</p>
                    {% endif %}
                </div>
            </div>
            """
            return self.render_string(widget_template, {
                'widget_type': widget_type,
                'widget_id': widget_data.get('widget_id', 'unknown'),
                'title': widget_data.get('title', 'Widget'),
                'data': widget_data.get('data'),
                'config': widget_data.get('config', {})
            })
        
        return self.render_template(template_name, {
            'widget': widget_data,
            'widget_type': widget_type
        })
    
    def render_agent_card(self, agent_data: Dict[str, Any]) -> str:
        """Render an agent information card."""
        agent_template = """
        <div class="card agent-card mb-3">
            <div class="card-body">
                <div class="d-flex justify-content-between align-items-start">
                    <div>
                        <h6 class="card-title">
                            <i class="{{ get_agent_type_icon(agent.agent_type) }}"></i>
                            {{ agent.name }}
                        </h6>
                        <p class="card-text text-muted">{{ agent.agent_type | title }} Agent</p>
                    </div>
                    <span class="badge {{ get_status_class(agent.status) }}">
                        {{ agent.status | title }}
                    </span>
                </div>
                
                <div class="row mt-3">
                    <div class="col-6">
                        <small class="text-muted">Tasks Completed</small>
                        <div class="fw-bold">{{ agent.total_tasks_completed }}</div>
                    </div>
                    <div class="col-6">
                        <small class="text-muted">Success Rate</small>
                        <div class="fw-bold">{{ agent.success_rate | percentage }}</div>
                    </div>
                </div>
                
                {% if agent.capabilities %}
                <div class="mt-3">
                    <small class="text-muted">Capabilities</small>
                    <div class="mt-1">
                        {% for capability in agent.capabilities %}
                        <span class="badge bg-light text-dark me-1">
                            <i class="{{ get_capability_icon(capability) }}"></i>
                            {{ capability | replace('_', ' ') | title }}
                        </span>
                        {% endfor %}
                    </div>
                </div>
                {% endif %}
            </div>
        </div>
        """
        
        return self.render_string(agent_template, {'agent': agent_data})
    
    def render_task_timeline_item(self, task_data: Dict[str, Any]) -> str:
        """Render a task timeline item."""
        timeline_template = """
        <div class="task-timeline-item">
            <div class="d-flex justify-content-between align-items-start">
                <div>
                    <strong>{{ task.description }}</strong>
                    {% if task.agent_name %}
                    <br><small class="text-muted">by {{ task.agent_name }}</small>
                    {% endif %}
                </div>
                <span class="badge {{ get_status_class(task.status) }}">
                    {{ task.status | title }}
                </span>
            </div>
            <small class="text-muted">{{ task.timestamp | datetime }}</small>
            {% if task.duration %}
            <small class="text-muted"> • {{ task.duration | duration }}</small>
            {% endif %}
        </div>
        """
        
        return self.render_string(timeline_template, {'task': task_data})


# Global template renderer instance
_template_renderer = None

def get_template_renderer(templates_dir: Optional[str] = None, logger: Optional[RunLogger] = None) -> TemplateRenderer:
    """Get or create the global template renderer instance."""
    global _template_renderer
    
    if _template_renderer is None:
        _template_renderer = TemplateRenderer(templates_dir, logger)
    
    return _template_renderer

def get_dashboard_renderer(templates_dir: Optional[str] = None, logger: Optional[RunLogger] = None) -> DashboardTemplateRenderer:
    """Get a dashboard-specific template renderer."""
    return DashboardTemplateRenderer(templates_dir, logger)
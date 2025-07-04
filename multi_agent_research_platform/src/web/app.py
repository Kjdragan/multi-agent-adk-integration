"""
Multi-Agent Research Platform Web Application

Main entry point for the ADK web interface providing debugging, monitoring,
and interaction capabilities for the multi-agent system.
"""

import asyncio
import signal
import sys
from typing import Optional
from pathlib import Path

from .config import WebConfig, DebugConfig, MonitoringConfig, WebSocketConfig, get_config_for_environment
from .interface import WebInterface
from .dashboards import AgentDashboard, TaskDashboard, PerformanceDashboard
from ..platform_logging import RunLogger, LogLevel
from ..services import SessionService, MemoryService, ArtifactService
from ..agents import AgentRegistry, AgentOrchestrator, AgentFactory, AgentSuite


class MultiAgentWebApp:
    """
    Main web application for the multi-agent research platform.
    
    Provides a complete web interface with debugging, monitoring, and
    interaction capabilities built on Google ADK.
    """
    
    def __init__(self, environment: str = "debug"):
        self.environment = environment
        self.config = get_config_for_environment(environment)
        
        # Initialize logger
        from ..platform_logging import LogConfig, LogLevel, LogFormat, setup_logging
        from pathlib import Path
        
        log_config = LogConfig(
            log_dir=Path("logs"),
            log_level=LogLevel.DEBUG if environment in ["debug", "development"] else LogLevel.INFO,
            log_format=LogFormat.STRUCTURED,
            max_log_files=100,
            log_retention_days=30
        )
        
        platform_logger = setup_logging(log_config)
        self.platform_logger = platform_logger
        
        # Create a simple logger for web app messages  
        import logging
        self.logger = logging.getLogger("web_app")
        self.logger.setLevel(log_config.log_level.value)
        
        # Initialize services
        self.session_service = SessionService()
        self.memory_service = MemoryService()
        self.artifact_service = ArtifactService()
        
        # Web interface
        self.web_interface = None
        
        # Dashboards
        self.dashboards = {}
        
        # Application state
        self.is_running = False
        self.shutdown_event = asyncio.Event()
        
        # Setup signal handlers
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating shutdown...")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def initialize(self) -> bool:
        """
        Initialize the web application.
        
        Returns:
            Success status
        """
        try:
            self.logger.info(f"Initializing Multi-Agent Web App in {self.environment} mode...")
            
            # Initialize services
            await self._initialize_services()
            
            # Initialize web interface
            await self._initialize_web_interface()
            
            # Initialize dashboards
            await self._initialize_dashboards()
            
            # Setup demo agents if in development/debug mode
            if self.environment in ["debug", "development"]:
                await self._setup_demo_agents()
            
            self.logger.info("Multi-Agent Web App initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize web application: {e}")
            return False
    
    async def _initialize_services(self):
        """Initialize core services."""
        try:
            # Initialize session service
            await self.session_service.initialize()
            self.logger.debug("Session service initialized")
            
            # Initialize memory service
            await self.memory_service.initialize()
            self.logger.debug("Memory service initialized")
            
            # Initialize artifact service
            await self.artifact_service.initialize()
            self.logger.debug("Artifact service initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize services: {e}")
            raise
    
    async def _initialize_web_interface(self):
        """Initialize the web interface."""
        try:
            self.web_interface = WebInterface(
                web_config=self.config["web"],
                debug_config=self.config["debug"],
                monitoring_config=self.config["monitoring"],
                websocket_config=self.config["websocket"],
                logger=self.logger,
                session_service=self.session_service,
                memory_service=self.memory_service,
                artifact_service=self.artifact_service
            )
            
            self.logger.debug("Web interface initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize web interface: {e}")
            raise
    
    async def _initialize_dashboards(self):
        """Initialize monitoring dashboards."""
        try:
            # Agent dashboard
            self.dashboards["agents"] = AgentDashboard(logger=self.logger)
            
            # Task dashboard (needs orchestrator)
            if hasattr(self, 'orchestrator'):
                self.dashboards["tasks"] = TaskDashboard(
                    orchestrator=self.orchestrator,
                    logger=self.logger
                )
            
            # Performance dashboard
            self.dashboards["performance"] = PerformanceDashboard(
                monitoring_interface=self.web_interface.monitoring_interface,
                logger=self.logger
            )
            
            self.logger.debug("Dashboards initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize dashboards: {e}")
            raise
    
    async def _setup_demo_agents(self):
        """Setup demo agents for development/debugging."""
        try:
            self.logger.info("Setting up demo agents...")
            
            # Create agent factory
            factory = AgentFactory(
                logger=self.logger,
                session_service=self.session_service
            )
            
            # Create research team
            research_agents = factory.create_agent_suite(
                suite_type=AgentSuite.RESEARCH_TEAM,
                custom_configs={
                    "researcher_domain": "AI/ML",
                    "analyst_focus": "technical analysis"
                }
            )
            
            # Activate agents
            for agent in research_agents:
                await agent.activate()
            
            # Create orchestrator
            self.orchestrator = AgentOrchestrator(
                logger=self.logger,
                session_service=self.session_service
            )
            
            # Update task dashboard with orchestrator
            self.dashboards["tasks"] = TaskDashboard(
                orchestrator=self.orchestrator,
                logger=self.logger
            )
            
            self.logger.info(f"Demo setup complete: {len(research_agents)} agents created")
            
        except Exception as e:
            self.logger.warning(f"Failed to setup demo agents: {e}")
    
    async def start(self) -> bool:
        """
        Start the web application.
        
        Returns:
            Success status
        """
        try:
            if self.is_running:
                self.logger.warning("Web application is already running")
                return True
            
            self.logger.info("Starting Multi-Agent Web App...")
            
            # Start web interface
            success = await self.web_interface.start()
            if not success:
                return False
            
            # Start dashboard refresh tasks
            await self._start_dashboard_tasks()
            
            self.is_running = True
            
            # Log access information
            web_config = self.config["web"]
            self.logger.info(f"🚀 Multi-Agent Web App started successfully!")
            self.logger.info(f"📊 Web Interface: http://{web_config.host}:{web_config.port}")
            self.logger.info(f"🔧 Debug Mode: {web_config.debug}")
            self.logger.info(f"🌐 WebSocket: ws://{web_config.host}:{web_config.port}/ws")
            self.logger.info(f"📈 Monitoring: {self.config['monitoring'].enable_real_time_updates}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start web application: {e}")
            return False
    
    async def _start_dashboard_tasks(self):
        """Start background tasks for dashboard updates."""
        try:
            for dashboard_name, dashboard in self.dashboards.items():
                if dashboard.auto_refresh_enabled:
                    task = asyncio.create_task(
                        self._dashboard_refresh_loop(dashboard_name, dashboard)
                    )
                    # Store task reference to prevent garbage collection
                    setattr(self, f"_{dashboard_name}_refresh_task", task)
            
            self.logger.debug("Dashboard refresh tasks started")
            
        except Exception as e:
            self.logger.error(f"Failed to start dashboard tasks: {e}")
    
    async def _dashboard_refresh_loop(self, dashboard_name: str, dashboard):
        """Background loop for dashboard refresh."""
        while self.is_running:
            try:
                await dashboard.refresh()
                await asyncio.sleep(dashboard.refresh_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Dashboard {dashboard_name} refresh error: {e}")
                await asyncio.sleep(dashboard.refresh_interval * 2)  # Longer delay on error
    
    async def stop(self) -> bool:
        """
        Stop the web application.
        
        Returns:
            Success status
        """
        try:
            if not self.is_running:
                self.logger.warning("Web application is not running")
                return True
            
            self.logger.info("Stopping Multi-Agent Web App...")
            
            self.is_running = False
            
            # Cancel dashboard tasks
            for attr_name in dir(self):
                if attr_name.endswith('_refresh_task'):
                    task = getattr(self, attr_name)
                    if task and not task.done():
                        task.cancel()
            
            # Stop web interface
            if self.web_interface:
                await self.web_interface.stop()
            
            # Stop services
            if self.session_service:
                await self.session_service.shutdown()
            
            if self.memory_service:
                await self.memory_service.shutdown()
            
            if self.artifact_service:
                await self.artifact_service.shutdown()
            
            self.logger.info("Multi-Agent Web App stopped successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop web application: {e}")
            return False
    
    async def shutdown(self):
        """Graceful shutdown of the application."""
        await self.stop()
        self.shutdown_event.set()
    
    async def run(self):
        """Run the web application until shutdown."""
        # Initialize
        success = await self.initialize()
        if not success:
            self.logger.error("Failed to initialize, exiting...")
            return
        
        # Start
        success = await self.start()
        if not success:
            self.logger.error("Failed to start, exiting...")
            return
        
        try:
            # Wait for shutdown signal
            await self.shutdown_event.wait()
        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt")
        finally:
            await self.stop()
    
    def get_status(self):
        """Get application status."""
        return {
            "is_running": self.is_running,
            "environment": self.environment,
            "web_interface": self.web_interface.get_status() if self.web_interface else None,
            "dashboards": {
                name: dashboard.get_dashboard_data() 
                for name, dashboard in self.dashboards.items()
            },
            "services": {
                "session_service": self.session_service.get_status() if self.session_service else None,
                "memory_service": self.memory_service.get_status() if self.memory_service else None,
                "artifact_service": self.artifact_service.get_status() if self.artifact_service else None,
            },
            "agents": AgentRegistry.get_registry_status(),
        }


async def main():
    """Main entry point for the web application."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-Agent Research Platform Web Interface")
    parser.add_argument(
        "--environment", "-e",
        choices=["development", "debug", "production", "monitoring"],
        default="debug",
        help="Environment configuration to use"
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Host to bind to (overrides config)"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=None,
        help="Port to bind to (overrides config)"
    )
    
    args = parser.parse_args()
    
    # Create and run application
    app = MultiAgentWebApp(environment=args.environment)
    
    # Override config if specified
    if args.host:
        app.config["web"].host = args.host
    if args.port:
        app.config["web"].port = args.port
    
    try:
        await app.run()
    except Exception as e:
        print(f"Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
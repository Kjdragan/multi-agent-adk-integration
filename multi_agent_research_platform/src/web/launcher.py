#!/usr/bin/env python3
"""
Web Interface Launcher

Quick launcher script for the Multi-Agent Research Platform web interface.
Provides easy startup with different configurations and environments.
"""

import asyncio
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.web.app import MultiAgentWebApp


def create_argument_parser():
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Research Platform Web Interface Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launcher.py                           # Run in debug mode (default)
  python launcher.py -e development           # Development mode
  python launcher.py -e production            # Production mode  
  python launcher.py -p 8080                  # Custom port
  python launcher.py --host 0.0.0.0 -p 8081   # Custom host and port
  python launcher.py -e monitoring            # Monitoring-optimized mode
        """
    )
    
    parser.add_argument(
        "--environment", "-e",
        choices=["development", "debug", "production", "monitoring"],
        default="debug",
        help="Environment configuration (default: debug)"
    )
    
    parser.add_argument(
        "--host",
        default=None,
        help="Host to bind to (overrides config default)"
    )
    
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=None,
        help="Port to bind to (overrides config default)"
    )
    
    parser.add_argument(
        "--no-demo",
        action="store_true",
        help="Skip demo agent creation"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Override log level"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Minimal console output"
    )
    
    parser.add_argument(
        "--version", "-v",
        action="version",
        version="Multi-Agent Research Platform 1.0.0"
    )
    
    return parser


def print_banner():
    """Print startup banner."""
    banner = """
    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │        🤖 Multi-Agent Research Platform Web Interface       │
    │                                                             │
    │        Built with Google Agent Development Kit (ADK)       │
    │        Debugging • Monitoring • Orchestration              │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
    """
    print(banner)


def print_startup_info(app: MultiAgentWebApp):
    """Print startup information and instructions."""
    config = app.config["web"]
    
    print("\n📋 Configuration:")
    print(f"   Environment: {app.environment}")
    print(f"   Host: {config.host}")
    print(f"   Port: {config.port}")
    print(f"   Debug Mode: {config.debug}")
    print(f"   WebSocket: {app.config['websocket'].enabled}")
    print(f"   Real-time Monitoring: {app.config['monitoring'].enable_real_time_updates}")
    
    print("\n🔗 Access URLs:")
    print(f"   Dashboard: http://{config.host}:{config.port}")
    print(f"   API Documentation: http://{config.host}:{config.port}/docs")
    print(f"   API Redoc: http://{config.host}:{config.port}/redoc")
    print(f"   Health Check: http://{config.host}:{config.port}/health")
    print(f"   System Status: http://{config.host}:{config.port}/status")
    
    if app.config['websocket'].enabled:
        print(f"   WebSocket: ws://{config.host}:{config.port}/ws")
    
    print("\n💡 Quick Actions:")
    print("   • Open the dashboard in your browser")
    print("   • Create agents via the web interface")
    print("   • Run orchestrated tasks and monitor progress")
    print("   • Use debug console for step-by-step debugging")
    print("   • View real-time logs and performance metrics")
    
    print("\n⚠️  Stop the server:")
    print("   Press Ctrl+C to gracefully shutdown")
    print()


async def main():
    """Main launcher function."""
    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Print banner unless quiet
    if not args.quiet:
        print_banner()
    
    try:
        # Create application
        app = MultiAgentWebApp(environment=args.environment)
        
        # Override configuration if specified
        if args.host:
            app.config["web"].host = args.host
        if args.port:
            app.config["web"].port = args.port
        if args.log_level:
            app.logger.level = args.log_level
        
        # Skip demo setup if requested
        if args.no_demo:
            app._setup_demo_agents = lambda: None
        
        # Print startup info unless quiet
        if not args.quiet:
            print_startup_info(app)
        
        # Initialize and run
        print("🚀 Starting Multi-Agent Web Application...")
        
        success = await app.initialize()
        if not success:
            print("❌ Failed to initialize application")
            return 1
        
        success = await app.start()
        if not success:
            print("❌ Failed to start application")
            return 1
        
        print("✅ Application started successfully!")
        print("   Ready to serve requests...")
        
        # Wait for shutdown
        try:
            await app.shutdown_event.wait()
        except KeyboardInterrupt:
            print("\n🛑 Shutdown signal received...")
        
        # Graceful shutdown
        print("⏳ Stopping application...")
        await app.stop()
        print("✅ Application stopped successfully")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        return 1
        
    except Exception as e:
        print(f"❌ Application error: {e}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        return 1


def run_launcher():
    """Entry point for the launcher."""
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Interrupted")
        sys.exit(1)


if __name__ == "__main__":
    run_launcher()
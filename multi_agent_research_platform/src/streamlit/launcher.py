#!/usr/bin/env python3
"""
Streamlit App Launcher

Launcher script for the Multi-Agent Research Platform Streamlit interface.
Provides easy startup with different configurations and environments.
"""

import subprocess
import sys
import argparse
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def create_argument_parser():
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Research Platform Streamlit Interface Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launcher.py                           # Run in production mode (default)
  python launcher.py -e development           # Development mode with debug features
  python launcher.py -e demo                  # Demo mode with sample data
  python launcher.py -p 8502                  # Custom port
  python launcher.py --host 0.0.0.0           # Custom host
  python launcher.py -e development --reload  # Auto-reload on file changes
        """
    )
    
    parser.add_argument(
        "--environment", "-e",
        choices=["development", "production", "demo", "minimal"],
        default="production",
        help="Environment configuration (default: production)"
    )
    
    parser.add_argument(
        "--host",
        default="localhost",
        help="Host to bind to (default: localhost)"
    )
    
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8501,
        help="Port to bind to (default: 8501)"
    )
    
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload on file changes (development only)"
    )
    
    parser.add_argument(
        "--browser",
        action="store_true",
        default=True,
        help="Open browser automatically (default: True)"
    )
    
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open browser automatically"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error"],
        default="info",
        help="Logging level (default: info)"
    )
    
    parser.add_argument(
        "--theme",
        choices=["light", "dark", "auto"],
        default=None,
        help="Override theme setting"
    )
    
    parser.add_argument(
        "--config-file",
        type=str,
        help="Custom configuration file path"
    )
    
    parser.add_argument(
        "--version", "-v",
        action="version",
        version="Multi-Agent Research Platform Streamlit Interface 1.0.0"
    )
    
    return parser


def print_banner():
    """Print startup banner."""
    banner = """
    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │    🤖 Multi-Agent Research Platform - Streamlit Interface   │
    │                                                             │
    │             Production-Ready User Interface                 │
    │         Research • Collaboration • Visualization           │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
    """
    print(banner)


def setup_environment_variables(args):
    """Setup environment variables for Streamlit."""
    # Set Streamlit configuration via environment variables
    env_vars = {
        "STREAMLIT_SERVER_HOST": args.host,
        "STREAMLIT_SERVER_PORT": str(args.port),
        "STREAMLIT_LOGGER_LEVEL": args.log_level.upper(),
        "STREAMLIT_CLIENT_TOOLBAR_MODE": "minimal",
        "STREAMLIT_SERVER_ENABLE_CORS": "true",
        "STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION": "true",
    }
    
    # Environment-specific settings
    if args.environment == "development":
        env_vars.update({
            "STREAMLIT_SERVER_RUN_ON_SAVE": "true" if args.reload else "false",
            "STREAMLIT_SERVER_FILE_WATCHER_TYPE": "auto",
            "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
        })
    elif args.environment == "production":
        env_vars.update({
            "STREAMLIT_SERVER_RUN_ON_SAVE": "false",
            "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
            "STREAMLIT_SERVER_ENABLE_STATIC_SERVING": "true",
        })
    
    # Theme setting
    if args.theme:
        env_vars["STREAMLIT_THEME_BASE"] = args.theme
    
    # Browser setting
    if args.no_browser:
        env_vars["STREAMLIT_SERVER_HEADLESS"] = "true"
    else:
        env_vars["STREAMLIT_SERVER_HEADLESS"] = "false"
    
    # Apply environment variables
    for key, value in env_vars.items():
        os.environ[key] = value


def print_startup_info(args):
    """Print startup information."""
    print("\n📋 Configuration:")
    print(f"   Environment: {args.environment}")
    print(f"   Host: {args.host}")
    print(f"   Port: {args.port}")
    print(f"   Log Level: {args.log_level}")
    print(f"   Auto-reload: {args.reload}")
    print(f"   Open Browser: {not args.no_browser}")
    
    print("\n🔗 Access URL:")
    print(f"   Application: http://{args.host}:{args.port}")
    
    print("\n💡 Features Available:")
    if args.environment == "development":
        print("   • Debug mode with detailed logging")
        print("   • Auto-reload on file changes")
        print("   • Development tools and metrics")
        print("   • Demo agents auto-created")
    elif args.environment == "demo":
        print("   • Demo mode with sample data")
        print("   • Extended analytics and charts")
        print("   • Pre-configured agent teams")
    elif args.environment == "minimal":
        print("   • Minimal interface with core features")
        print("   • Reduced resource usage")
        print("   • Basic agent and task management")
    else:  # production
        print("   • Production-optimized interface")
        print("   • Enhanced security and rate limiting")
        print("   • Caching and performance optimizations")
        print("   • Privacy-focused configuration")
    
    print("\n📚 Quick Start:")
    print("   1. Create agents using the sidebar")
    print("   2. Enter a research question")
    print("   3. Choose orchestration strategy") 
    print("   4. Execute and view results")
    print("   5. Explore analytics and history")
    
    print("\n⚠️  Stop the server:")
    print("   Press Ctrl+C in this terminal")
    print()


def run_streamlit_app(args):
    """Run the Streamlit application."""
    try:
        # Get the path to the Streamlit app
        app_path = Path(__file__).parent / "app.py"
        
        # Build streamlit command
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            str(app_path),
            "--server.address", args.host,
            "--server.port", str(args.port),
            "--logger.level", args.log_level,
        ]
        
        # Add environment-specific arguments
        if args.environment == "development" and args.reload:
            cmd.extend(["--server.runOnSave", "true"])
        
        if args.no_browser:
            cmd.extend(["--server.headless", "true"])
        
        # Add theme if specified
        if args.theme:
            cmd.extend(["--theme.base", args.theme])
        
        # Set environment variable for the app to know its configuration
        os.environ["STREAMLIT_ENVIRONMENT"] = args.environment
        
        print("🚀 Starting Streamlit application...")
        print(f"   Command: {' '.join(cmd)}")
        print("   This may take a moment to initialize...")
        print()
        
        # Run Streamlit
        subprocess.run(cmd, check=True)
        
    except KeyboardInterrupt:
        print("\n🛑 Streamlit application stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start Streamlit application: {e}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1
    
    return 0


def check_dependencies():
    """Check if required dependencies are available."""
    required_packages = [
        "streamlit",
        "plotly", 
        "pandas",
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Install missing packages:")
        print(f"   pip install {' '.join(missing_packages)}")
        print("   or")
        print("   uv add streamlit plotly pandas")
        return False
    
    return True


def main():
    """Main launcher function."""
    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Print startup info
    print_startup_info(args)
    
    # Setup environment
    setup_environment_variables(args)
    
    # Run the application
    return run_streamlit_app(args)


def run_launcher():
    """Entry point for the launcher."""
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Interrupted")
        sys.exit(1)


if __name__ == "__main__":
    run_launcher()
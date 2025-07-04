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

# Import centralized configuration
from src.config.manager import validate_startup_configuration, get_config_manager


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


def setup_streamlit_environment(args):
    """Setup minimal Streamlit-specific environment variables."""
    # Only set Streamlit-specific variables that don't conflict with app configuration
    # Use command-line arguments instead of environment variables where possible
    
    # Set only essential Streamlit configuration that can't be passed via CLI
    streamlit_env = {
        "STREAMLIT_CLIENT_TOOLBAR_MODE": "minimal",
        "STREAMLIT_SERVER_ENABLE_CORS": "true", 
        "STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION": "true",
        "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
    }
    
    # Environment-specific Streamlit settings (minimal)
    if args.environment == "development":
        streamlit_env["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "auto"
    elif args.environment == "production":
        streamlit_env["STREAMLIT_SERVER_ENABLE_STATIC_SERVING"] = "true"
    
    # Apply only Streamlit-specific environment variables
    for key, value in streamlit_env.items():
        os.environ[key] = value
    
    # Note: Host, port, theme, etc. are now passed via command line arguments
    # This avoids conflicts with the centralized configuration system


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
        
        # Configuration is now handled by centralized configuration manager
        # No need to set STREAMLIT_ENVIRONMENT manually
        
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
    
    # Validate configuration early to catch issues
    print("🔧 Validating configuration...")
    try:
        validate_startup_configuration()
        print("✅ Configuration validation passed")
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        print("\nPlease check your .env file and ensure all required settings are configured.")
        return 1
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Print startup info
    print_startup_info(args)
    
    # Setup Streamlit-specific environment (minimal, non-conflicting)
    setup_streamlit_environment(args)
    
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
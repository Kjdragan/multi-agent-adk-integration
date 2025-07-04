#!/usr/bin/env python3
"""
Test runner script for Multi-Agent Research Platform
Provides convenient commands for running different types of tests
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


class TestRunner:
    """Main test runner class."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.tests_dir = self.project_root / "tests"
        
    def run_command(self, command: List[str], description: str) -> int:
        """Run a command and return exit code."""
        print(f"\n🚀 {description}")
        print(f"Running: {' '.join(command)}")
        print("-" * 60)
        
        # Set up environment for proper imports
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.project_root)
        
        try:
            result = subprocess.run(command, cwd=self.project_root, env=env, check=False)
            return result.returncode
        except KeyboardInterrupt:
            print("\n❌ Test execution interrupted by user")
            return 130
        except Exception as e:
            print(f"\n❌ Error running command: {e}")
            return 1
    
    def run_unit_tests(self, verbose: bool = False, coverage: bool = True) -> int:
        """Run unit tests."""
        command = ["python", "-m", "pytest", "tests/unit/"]
        
        if verbose:
            command.extend(["-v", "-s"])
        
        if not coverage:
            command.extend(["--no-cov"])
        
        return self.run_command(command, "Running Unit Tests")
    
    def run_integration_tests(self, verbose: bool = False) -> int:
        """Run integration tests."""
        command = ["python", "-m", "pytest", "tests/integration/", "-m", "integration"]
        
        if verbose:
            command.extend(["-v", "-s"])
        
        # Skip if no API keys are available
        if not os.getenv("GOOGLE_API_KEY"):
            command.extend(["-m", "not requires_api_key"])
            print("⚠️  Skipping tests that require API keys (GOOGLE_API_KEY not set)")
        
        return self.run_command(command, "Running Integration Tests")
    
    def run_e2e_tests(self, verbose: bool = False) -> int:
        """Run end-to-end tests."""
        command = ["python", "-m", "pytest", "tests/e2e/", "-m", "e2e"]
        
        if verbose:
            command.extend(["-v", "-s"])
        
        # Skip if no API keys are available
        if not os.getenv("GOOGLE_API_KEY"):
            command.extend(["-m", "not requires_api_key"])
            print("⚠️  Skipping tests that require API keys (GOOGLE_API_KEY not set)")
        
        return self.run_command(command, "Running End-to-End Tests")
    
    def run_performance_tests(self, verbose: bool = False) -> int:
        """Run performance tests."""
        command = ["python", "-m", "pytest", "tests/performance/", "-m", "performance"]
        
        if verbose:
            command.extend(["-v", "-s"])
        
        # Performance tests are slow by nature
        command.extend(["--timeout=600"])  # 10 minutes timeout
        
        return self.run_command(command, "Running Performance Tests")
    
    def run_smoke_tests(self, verbose: bool = False) -> int:
        """Run quick smoke tests."""
        command = ["python", "-m", "pytest", "-m", "smoke"]
        
        if verbose:
            command.extend(["-v", "-s"])
        
        # Fast execution
        command.extend(["--timeout=60", "--maxfail=5"])
        
        return self.run_command(command, "Running Smoke Tests")
    
    def run_all_tests(self, verbose: bool = False, skip_slow: bool = False) -> int:
        """Run all tests in sequence."""
        tests_to_run = [
            ("Unit Tests", lambda: self.run_unit_tests(verbose)),
            ("Integration Tests", lambda: self.run_integration_tests(verbose)),
            ("End-to-End Tests", lambda: self.run_e2e_tests(verbose)),
        ]
        
        if not skip_slow:
            tests_to_run.append(("Performance Tests", lambda: self.run_performance_tests(verbose)))
        
        results = {}
        total_failed = 0
        
        for test_name, test_func in tests_to_run:
            print(f"\n{'='*60}")
            print(f"Starting {test_name}")
            print(f"{'='*60}")
            
            exit_code = test_func()
            results[test_name] = exit_code
            
            if exit_code != 0:
                total_failed += 1
                print(f"❌ {test_name} FAILED (exit code: {exit_code})")
            else:
                print(f"✅ {test_name} PASSED")
        
        # Print summary
        print(f"\n{'='*60}")
        print("TEST EXECUTION SUMMARY")
        print(f"{'='*60}")
        
        for test_name, exit_code in results.items():
            status = "✅ PASSED" if exit_code == 0 else "❌ FAILED"
            print(f"{test_name:<20} {status}")
        
        print(f"\nTotal: {len(results)} test suites, {total_failed} failed")
        
        return total_failed
    
    def run_specific_test(self, test_path: str, verbose: bool = False) -> int:
        """Run a specific test file or directory."""
        command = ["python", "-m", "pytest", test_path]
        
        if verbose:
            command.extend(["-v", "-s"])
        
        return self.run_command(command, f"Running Specific Test: {test_path}")
    
    def run_with_markers(self, markers: List[str], verbose: bool = False) -> int:
        """Run tests with specific markers."""
        command = ["python", "-m", "pytest"]
        
        for marker in markers:
            command.extend(["-m", marker])
        
        if verbose:
            command.extend(["-v", "-s"])
        
        return self.run_command(command, f"Running Tests with Markers: {', '.join(markers)}")
    
    def run_coverage_report(self) -> int:
        """Generate detailed coverage report."""
        commands = [
            (["python", "-m", "pytest", "--cov=src", "--cov-report=html", "--cov-report=term"], 
             "Generating Coverage Report"),
            (["python", "-c", "import webbrowser; webbrowser.open('htmlcov/index.html')"], 
             "Opening Coverage Report in Browser")
        ]
        
        for command, description in commands:
            exit_code = self.run_command(command, description)
            if exit_code != 0:
                return exit_code
        
        return 0
    
    def run_linting(self) -> int:
        """Run code linting and formatting checks."""
        commands = [
            (["python", "-m", "flake8", "src/", "tests/"], "Running Flake8 Linting"),
            (["python", "-m", "black", "--check", "src/", "tests/"], "Checking Black Formatting"),
            (["python", "-m", "isort", "--check-only", "src/", "tests/"], "Checking Import Sorting"),
            (["python", "-m", "mypy", "src/"], "Running Type Checking")
        ]
        
        failed_checks = 0
        
        for command, description in commands:
            exit_code = self.run_command(command, description)
            if exit_code != 0:
                failed_checks += 1
        
        if failed_checks > 0:
            print(f"\n❌ {failed_checks} linting checks failed")
        else:
            print("\n✅ All linting checks passed")
        
        return failed_checks
    
    def setup_test_environment(self) -> int:
        """Set up test environment."""
        commands = [
            (["python", "-m", "pip", "install", "-r", "requirements.txt"], "Installing Dependencies"),
            (["python", "-m", "pip", "install", "pytest", "pytest-asyncio", "pytest-cov", "pytest-timeout"], 
             "Installing Test Dependencies")
        ]
        
        for command, description in commands:
            exit_code = self.run_command(command, description)
            if exit_code != 0:
                return exit_code
        
        print("\n✅ Test environment set up successfully")
        return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test runner for Multi-Agent Research Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py unit                    # Run unit tests
  python run_tests.py integration -v          # Run integration tests with verbose output
  python run_tests.py all --skip-slow         # Run all tests except slow ones
  python run_tests.py specific tests/unit/test_agents.py  # Run specific test file
  python run_tests.py markers unit mock_apis  # Run tests with specific markers
  python run_tests.py coverage                # Generate coverage report
  python run_tests.py lint                    # Run linting checks
  python run_tests.py setup                   # Set up test environment
        """
    )
    
    parser.add_argument(
        "command",
        choices=["unit", "integration", "e2e", "performance", "smoke", "all", "specific", "markers", "coverage", "lint", "setup"],
        help="Test command to run"
    )
    
    parser.add_argument(
        "args",
        nargs="*",
        help="Additional arguments (for 'specific' or 'markers' commands)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--skip-slow",
        action="store_true",
        help="Skip slow-running tests (only for 'all' command)"
    )
    
    parser.add_argument(
        "--no-coverage",
        action="store_true",
        help="Disable coverage reporting (only for 'unit' command)"
    )
    
    args = parser.parse_args()
    
    runner = TestRunner()
    
    # Ensure we're in the correct directory
    os.chdir(runner.project_root)
    
    try:
        if args.command == "unit":
            exit_code = runner.run_unit_tests(args.verbose, not args.no_coverage)
        elif args.command == "integration":
            exit_code = runner.run_integration_tests(args.verbose)
        elif args.command == "e2e":
            exit_code = runner.run_e2e_tests(args.verbose)
        elif args.command == "performance":
            exit_code = runner.run_performance_tests(args.verbose)
        elif args.command == "smoke":
            exit_code = runner.run_smoke_tests(args.verbose)
        elif args.command == "all":
            exit_code = runner.run_all_tests(args.verbose, args.skip_slow)
        elif args.command == "specific":
            if not args.args:
                print("❌ Error: 'specific' command requires a test path")
                return 1
            exit_code = runner.run_specific_test(args.args[0], args.verbose)
        elif args.command == "markers":
            if not args.args:
                print("❌ Error: 'markers' command requires at least one marker")
                return 1
            exit_code = runner.run_with_markers(args.args, args.verbose)
        elif args.command == "coverage":
            exit_code = runner.run_coverage_report()
        elif args.command == "lint":
            exit_code = runner.run_linting()
        elif args.command == "setup":
            exit_code = runner.setup_test_environment()
        else:
            print(f"❌ Unknown command: {args.command}")
            return 1
        
        return exit_code
    
    except KeyboardInterrupt:
        print("\n❌ Test execution interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
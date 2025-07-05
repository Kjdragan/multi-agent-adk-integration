#!/usr/bin/env python3
"""
Start a debugging session with comprehensive logging
"""
import os
import sys
import subprocess
import signal
import time
from pathlib import Path

def setup_debug_environment():
    """Set up environment variables for maximum logging visibility."""
    debug_env = {
        'LOG_LEVEL': 'DEBUG',
        'PYTHONUNBUFFERED': '1',  # Ensure immediate output
        'STREAMLIT_SERVER_ENABLECORS': 'false',
        'STREAMLIT_SERVER_ENABLEXSRFPROTECTION': 'false',
        'GOOGLE_GENAI_USE_VERTEXAI': 'True',
        'GOOGLE_CLOUD_PROJECT': 'cs-poc-czxf7xbmmrua9yw8mrrkrn0'
    }
    
    for key, value in debug_env.items():
        os.environ[key] = value
        print(f"Set {key}={value}")

def start_log_monitor():
    """Start the log monitoring process."""
    print("🔍 Starting log monitor...")
    return subprocess.Popen([
        sys.executable, 'debug_logs.py'
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def start_streamlit():
    """Start Streamlit with debug settings."""
    print("🚀 Starting Streamlit with debug logging...")
    return subprocess.Popen([
        sys.executable, '-m', 'src.streamlit.launcher',
        '-e', 'development',
        '--port', '8504'
    ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

def main():
    print("🐛 MULTI-AGENT PLATFORM DEBUG SESSION")
    print("="*60)
    
    # Setup environment
    setup_debug_environment()
    print()
    
    processes = []
    
    try:
        # Start log monitor
        log_monitor = start_log_monitor()
        processes.append(log_monitor)
        time.sleep(1)
        
        # Start Streamlit
        streamlit_proc = start_streamlit()
        processes.append(streamlit_proc)
        
        print()
        print("✅ Debug session started!")
        print("📊 Streamlit: http://localhost:8504")
        print("🔍 Log monitor: Running in background")
        print()
        print("🚨 IMPORTANT DEBUGGING INSTRUCTIONS:")
        print("1. Open http://localhost:8504 in your browser")
        print("2. Try running a simple task like 'What is 2+2?'")
        print("3. Watch this terminal for detailed logs")
        print("4. All task execution will be logged in detail")
        print("5. Press Ctrl+C to stop all processes")
        print()
        print("🔍 Waiting for Streamlit startup...")
        
        # Monitor Streamlit output and relay to console
        while True:
            output = streamlit_proc.stdout.readline()
            if output:
                print(f"STREAMLIT: {output.strip()}")
                
                # Check if Streamlit is ready
                if "You can now view your Streamlit app" in output:
                    print("\n🎉 Streamlit is ready!")
                    print("👆 Now go to your browser and test a task")
                    print("📝 All logs will appear below:")
                    print("-" * 60)
                    break
            
            if streamlit_proc.poll() is not None:
                print("❌ Streamlit process ended unexpectedly")
                break
                
            time.sleep(0.1)
        
        # Continue monitoring until interrupted
        while True:
            # Check if Streamlit output
            output = streamlit_proc.stdout.readline()
            if output:
                print(f"STREAMLIT: {output.strip()}")
            
            # Check if processes are still running
            if streamlit_proc.poll() is not None:
                print("❌ Streamlit stopped")
                break
                
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping debug session...")
    except Exception as e:
        print(f"❌ Error in debug session: {e}")
    finally:
        # Clean up processes
        for proc in processes:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except:
                proc.kill()
        
        print("✅ Debug session stopped")

if __name__ == "__main__":
    main()
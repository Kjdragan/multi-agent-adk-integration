#!/usr/bin/env python3
"""
Real-time log monitoring script for debugging Streamlit task execution
"""
import time
import os
from pathlib import Path
from datetime import datetime

def monitor_logs():
    """Monitor logs in real-time."""
    log_dir = Path("logs")
    
    print("🔍 STREAMLIT TASK EXECUTION LOG MONITOR")
    print("="*60)
    print(f"Monitoring directory: {log_dir.absolute()}")
    print(f"Started at: {datetime.now()}")
    print("="*60)
    print()
    
    # Monitor files
    files_to_monitor = [
        log_dir / "global_errors.log",
        log_dir / "debug.log",
        log_dir / "task_execution.log"
    ]
    
    # Track file positions
    file_positions = {}
    
    while True:
        try:
            for log_file in files_to_monitor:
                if log_file.exists():
                    # Get current file size
                    current_size = log_file.stat().st_size
                    last_position = file_positions.get(str(log_file), 0)
                    
                    if current_size > last_position:
                        # Read new content
                        with open(log_file, 'r') as f:
                            f.seek(last_position)
                            new_content = f.read()
                            
                            if new_content.strip():
                                print(f"\n📁 {log_file.name}:")
                                print("-" * 40)
                                print(new_content.strip())
                                print("-" * 40)
                        
                        file_positions[str(log_file)] = current_size
            
            # Also monitor any new run directories
            runs_dir = log_dir / "runs"
            if runs_dir.exists():
                for run_dir in runs_dir.iterdir():
                    if run_dir.is_dir():
                        for log_file in run_dir.glob("*.log"):
                            if log_file.exists():
                                current_size = log_file.stat().st_size
                                last_position = file_positions.get(str(log_file), 0)
                                
                                if current_size > last_position:
                                    with open(log_file, 'r') as f:
                                        f.seek(last_position)
                                        new_content = f.read()
                                        
                                        if new_content.strip():
                                            print(f"\n📁 {log_file.parent.name}/{log_file.name}:")
                                            print("-" * 40)
                                            print(new_content.strip())
                                            print("-" * 40)
                                    
                                    file_positions[str(log_file)] = current_size
            
            time.sleep(0.5)  # Check every 500ms
            
        except KeyboardInterrupt:
            print("\n\n🛑 Log monitoring stopped")
            break
        except Exception as e:
            print(f"Error monitoring logs: {e}")
            time.sleep(1)

if __name__ == "__main__":
    monitor_logs()
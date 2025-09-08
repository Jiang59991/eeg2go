#!/usr/bin/env python3
"""
Script to switch to local mode
"""

import os
import sys

def switch_to_local_mode():
    """Switch to local mode"""
    print("Switching to local mode...")
    
    # Set environment variables
    os.environ['USE_LOCAL_EXECUTOR'] = 'true'
    os.environ['LOCAL_EXECUTOR_WORKERS'] = '1'  # Local mode uses single thread
    
    print("Environment variables set:")
    print(f"  USE_LOCAL_EXECUTOR = {os.environ.get('USE_LOCAL_EXECUTOR', 'not set')}")
    print(f"  LOCAL_EXECUTOR_WORKERS = {os.environ.get('LOCAL_EXECUTOR_WORKERS', 'not set')}")
    
    # Start local system
    print("\nStarting local mode system...")
    os.system('python start_local_system.py')

def switch_to_celery_mode():
    """Switch to Celery mode"""
    print("Switching to Celery mode...")
    
    # Clear environment variables
    if 'USE_LOCAL_EXECUTOR' in os.environ:
        del os.environ['USE_LOCAL_EXECUTOR']
    if 'LOCAL_EXECUTOR_WORKERS' in os.environ:
        del os.environ['LOCAL_EXECUTOR_WORKERS']
    
    print("Environment variables cleared, will use default Celery mode")
    print("Please manually start Celery worker and web application")

def show_current_mode():
    """Show current mode"""
    use_local = os.environ.get('USE_LOCAL_EXECUTOR', 'false').lower() == 'true'
    mode = "Local Mode" if use_local else "Celery Mode"
    workers = os.environ.get('LOCAL_EXECUTOR_WORKERS', 'N/A')
    
    print(f"Current system mode: {mode}")
    print(f"Local executor worker threads: {workers}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python switch_to_local_mode.py local    # Switch to local mode")
        print("  python switch_to_local_mode.py celery   # Switch to Celery mode")
        print("  python switch_to_local_mode.py status   # Show current mode")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == 'local':
        switch_to_local_mode()
    elif command == 'celery':
        switch_to_celery_mode()
    elif command == 'status':
        show_current_mode()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)

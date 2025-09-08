#!/usr/bin/env python3
"""
Start local mode system
No Redis required, all tasks run locally
"""
import os
import sys
import signal
import time
from datetime import datetime

# Set environment variables to enable local mode
os.environ['USE_LOCAL_EXECUTOR'] = 'true'

# Add project root directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from logging_config import logger
from task_queue.local_executor import get_local_executor, shutdown_local_executor

def signal_handler(signum, frame):
    """Signal handler for graceful shutdown"""
    logger.info(f"Received signal {signum}, shutting down...")
    shutdown_local_executor()
    sys.exit(0)

def main():
    """Main function"""
    print("=" * 60)
    print("EEG2Go Local Mode System Starting")
    print("=" * 60)
    print(f"Start time: {datetime.now()}")
    print("Mode: Local Executor (No Redis)")
    print("=" * 60)
    
    # Set signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Initialize local executor
        logger.info("Initializing local executor...")
        local_executor = get_local_executor()
        
        print("Local executor started")
        print("System ready to receive tasks")
        print("Press Ctrl+C to exit")
        print("-" * 60)
        
        # Keep system running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"System error: {e}")
    finally:
        logger.info("Shutting down local system...")
        shutdown_local_executor()
        print("System shutdown complete")

if __name__ == "__main__":
    main()


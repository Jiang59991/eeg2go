#!/usr/bin/env python3
"""
EEG2Go Celery Worker Startup Script

Start Celery Worker processes. The web application needs to be started separately.
"""
import subprocess
import time
import os
import sys
import signal
import threading
from pathlib import Path

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from logging_config import logger

class SystemManager:
    def __init__(self, workers=4):
        self.workers = workers
        self.worker_processes = []
        self.running = False
        
    def start_workers(self):
        """Start Celery Worker processes"""
        logger.info(f"Starting {self.workers} Celery Worker processes...")
        
        # Start different types of workers with optimized concurrency
        worker_configs = [
            {'queue': 'feature_extraction', 'concurrency': 1, 'name': 'feature_worker'},
            {'queue': 'experiments', 'concurrency': 1, 'name': 'experiment_worker'},
            {'queue': 'recordings', 'concurrency': 1, 'name': 'recordings_worker_1'},
            {'queue': 'recordings', 'concurrency': 1, 'name': 'recordings_worker_2'}
            # {'queue': 'default', 'concurrency': 1, 'name': 'default_worker'}
        ]
        
        for i, config in enumerate(worker_configs[:self.workers]):
            env = os.environ.copy()
            env['DATABASE_PATH'] = 'database/eeg2go.db'
            
            cmd = [
                sys.executable,
                "task_queue/celery_worker.py",
                "--queue", config['queue'],
                "--concurrency", str(config['concurrency']),
                "--hostname", f"{config['name']}@{os.uname().nodename if hasattr(os, 'uname') else 'localhost'}"
            ]
            
            if os.name == 'nt':  # Windows
                process = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                    text=True
                )
            else:  # Unix/Linux
                process = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
            
            self.worker_processes.append({
                'process': process,
                'config': config,
                'id': i + 1
            })
            
            logger.info(f"Worker {i+1} ({config['queue']}) started, PID: {process.pid}")
            time.sleep(3)  # Increased start interval to reduce resource contention
    
    def start(self):
        """Start Celery Worker processes"""
        logger.info("=" * 50)
        logger.info("        EEG2Go Celery Worker Startup")
        logger.info("=" * 50)
        
        try:
            # Start worker processes
            self.start_workers()
            
            self.running = True
            logger.info("Celery Worker processes started successfully!")
            logger.info("Please manually start the web application: python web/scripts/start_web_interface.py")
            logger.info("Press Ctrl+C to stop the Worker processes...")
            
            # Wait for user interruption
            while self.running:
                time.sleep(1)
                
                # Check process status
                for worker_info in self.worker_processes:
                    process = worker_info['process']
                    config = worker_info['config']
                    worker_id = worker_info['id']
                    
                    if process.poll() is not None:
                        logger.error(f"Worker {worker_id} ({config['queue']}) has exited")
            
        except KeyboardInterrupt:
            logger.info("Stopping Celery Worker processes...")
        finally:
            self.stop()
    
    def stop(self):
        """Stop Celery Worker processes"""
        logger.info("Stopping all Celery Worker processes...")
        
        # Stop worker processes
        for worker_info in self.worker_processes:
            process = worker_info['process']
            config = worker_info['config']
            worker_id = worker_info['id']
            
            if process.poll() is None:
                logger.info(f"Stopping Worker {worker_id} ({config['queue']})...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
        
        logger.info("All Celery Worker processes have been stopped")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='EEG2Go Celery Worker Startup Script')
    parser.add_argument('--workers', type=int, default=4, help='Number of Worker processes')
    
    args = parser.parse_args()
    
    manager = SystemManager(args.workers)
    manager.start()

if __name__ == "__main__":
    main() 
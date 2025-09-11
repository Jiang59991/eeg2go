#!/usr/bin/env python3
import os
import sys
import subprocess

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

def setup_local_mode() -> bool:
    """
    Set local mode environment variables if USE_LOCAL_EXECUTOR is set to true.

    Returns:
        bool: True if local mode is enabled, False otherwise.
    """
    use_local = os.getenv('USE_LOCAL_EXECUTOR', 'false').lower() == 'true'
    if use_local:
        print("Local mode detected, enabling local executor...")
        os.environ['USE_LOCAL_EXECUTOR'] = 'true'
        os.environ['LOCAL_EXECUTOR_WORKERS'] = os.getenv('LOCAL_EXECUTOR_WORKERS', '1')
        return True
    else:
        print("Using default Celery mode...")
        return False

def check_dependencies() -> bool:
    """
    Check if required dependencies are installed.

    Returns:
        bool: True if all dependencies are installed, False otherwise.
    """
    required_packages = ['flask', 'pandas', 'numpy', 'mne']
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    return True

def check_database() -> bool:
    """
    Check if the database exists by importing DATABASE_PATH from web.config.

    Returns:
        bool: True if the database exists, False otherwise.
    """
    try:
        from web.config import DATABASE_PATH
        if not os.path.exists(DATABASE_PATH):
            print("Database not found. Please run the setup scripts first:")
            print("python main.py")
            return False
        return True
    except ImportError as e:
        print(f"Cannot import web.config: {e}")
        print("Please check the project structure.")
        return False

def main() -> None:
    """
    Main function to start the EEG2Go web interface.
    """
    print("EEG2Go Web Interface")
    print("=" * 30)
    is_local_mode = setup_local_mode()
    if not check_dependencies():
        sys.exit(1)
    if not check_database():
        sys.exit(1)
    if is_local_mode:
        print("Mode: Local Executor (No Redis)")
        print(f"Worker threads: {os.environ.get('LOCAL_EXECUTOR_WORKERS', '1')}")
    else:
        print("Mode: Celery Distributed (Using Redis)")
    print("Starting web interface...")
    print("Access the interface at: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("-" * 30)
    try:
        from web import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
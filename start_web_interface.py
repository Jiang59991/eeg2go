#!/usr/bin/env python3
"""
EEG2Go Web Interface Startup Script

This script starts the Flask web interface for EEG feature extraction.
"""

import os
import sys
import subprocess

def check_dependencies():
    """Check if required dependencies are installed"""
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

def check_database():
    """Check if database exists"""
    db_path = os.path.join(os.path.dirname(__file__), "database", "eeg2go.db")
    if not os.path.exists(db_path):
        print("Database not found. Please run the setup scripts first:")
        print("python main.py")
        return False
    
    return True

def main():
    """Main function"""
    print("EEG2Go Web Interface")
    print("=" * 30)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check database
    if not check_database():
        sys.exit(1)
    
    print("Starting web interface...")
    print("Access the interface at: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("-" * 30)
    
    # Start Flask app
    try:
        from web_interface import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
EEG2Go Web Interface Startup Script

This script starts the Flask web interface for EEG feature extraction.
"""

import os
import sys
import subprocess

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

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
    # 使用config中的数据库路径
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
        # 修正导入路径：从web包导入app
        from web import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
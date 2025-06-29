#!/usr/bin/env python3
"""
Startup script for Vision Agent server
"""
import os
import sys

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    from server.main import main
    main()
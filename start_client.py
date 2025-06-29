#!/usr/bin/env python3
"""
Startup script for Vision Agent desktop client
"""
import os
import sys
import asyncio

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    from desktop_agent.main import main
    
    # Check if we're on Windows and need to set the event loop policy
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    asyncio.run(main())
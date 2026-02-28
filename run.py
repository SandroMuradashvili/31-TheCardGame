#!/usr/bin/env python3
"""
BURA Game Launcher
Run this file to start the game!
"""
import subprocess
import sys
import os
import time
import webbrowser
import threading

def open_browser():
    time.sleep(1.5)
    webbrowser.open('http://localhost:5000')

if __name__ == '__main__':
    print("=" * 50)
    print("  BURA — Georgian Card Game")
    print("=" * 50)
    print()
    print("Starting server on http://localhost:5000")
    print("Press Ctrl+C to quit.")
    print()

    # Open browser in background
    t = threading.Thread(target=open_browser, daemon=True)
    t.start()

    # Run the server
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    from server import app
    app.run(debug=False, port=5000, use_reloader=False)
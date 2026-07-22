import os
import sys
import webbrowser
import time

# Ensure UTF-8 output encoding on Windows terminals to prevent UnicodeEncodeError
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='ignore')
        sys.stderr.reconfigure(encoding='utf-8', errors='ignore')
    except Exception:
        pass

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from web_app.server import start_server

def main():
    port = 8000
    url = f"http://localhost:{port}"
    print(f"Opening browser at {url} ...")
    webbrowser.open(url)
    start_server(port)

if __name__ == "__main__":
    main()

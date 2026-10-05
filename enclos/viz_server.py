#!/usr/bin/env python3
"""Serve the live enumeration dashboard on http://127.0.0.1:8765/"""

import functools
import http.server
import os
from pathlib import Path

DIR = Path("/workspace/enclos/viz")
os.chdir(DIR)

class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # quiet

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

if __name__ == "__main__":
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 8765), Handler)
    print("Serving viz at http://127.0.0.1:8765/", flush=True)
    server.serve_forever()

import multiprocessing
import socket
import threading
import time
import urllib.request
import webview
import uvicorn
import os
import sys

from api_main import app

def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

def run_server(port):
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="critical")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    port = get_free_port()
    url = f"http://127.0.0.1:{port}"
    
    # سرور در ترد جداگانه
    server_thread = threading.Thread(target=run_server, args=(port,), daemon=True)
    server_thread.start()
    
    # صبر تا سرور بیدار شود
    for _ in range(50):
        try:
            urllib.request.urlopen(url, timeout=0.5).close()
            break
        except OSError:
            time.sleep(0.1)
            
    # پنجره بومی ویندوز
    icon_path = os.path.join(getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__))), "icon.png")
    
    window = webview.create_window(
        title="Metric Studio \u2014 Image & Volume Similarity Analyzer",
        url=url,
        width=1200,
        height=850,
        min_size=(900, 600)
    )
    webview.start(private_mode=False)

"""Starts the web app on a free port and opens your browser.  Use START_APP.bat, or:  python run_app.py"""
import socket
import sys
import threading
import warnings
import webbrowser
from pathlib import Path

warnings.filterwarnings("ignore")          # hide harmless library notices so the window stays readable
try:                                       # a Windows console must never crash on a special character
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except Exception:
    pass
sys.path.insert(0, str(Path(__file__).parent))


def in_use(port: int) -> bool:
    """True if a program is already listening on that port. (A connect test is reliable on Windows;
    a bind test is not, because Windows lets two programs share one port.)"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.4)
        return s.connect_ex(("127.0.0.1", port)) == 0


def free_port(start=5050, tries=40) -> int:
    for port in range(start, start + tries):
        if not in_use(port):
            return port
    raise SystemExit(f"No free port between {start} and {start + tries}. Run STOP_ALL_PYTHON.bat and try again.")


if __name__ == "__main__":
    print("\n" + "=" * 66)
    print(" Data-Driven Machine Learning Recommender for Mechanical Material Selection")
    print("=" * 66)
    print("Loading your data and training the models (can take up to a minute) ...\n")
    import app as webapp
    webapp.load_all()
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    print("\n" + "=" * 66)
    print(f" READY.  Open this address in your browser:   {url}")
    print(" (your browser should open by itself)")
    print(" To stop the app: click this window and press Ctrl + C")
    print("=" * 66 + "\n")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    webapp.app.run(host="127.0.0.1", port=port, debug=False, threaded=True, use_reloader=False)

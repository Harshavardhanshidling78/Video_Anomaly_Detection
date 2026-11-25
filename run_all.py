import os
import sys
import time
import socket
import argparse
import webbrowser
import subprocess
import urllib.request
import urllib.error
from pathlib import Path

# Resolve Project Root Directory
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def find_python_executable() -> str:
    """
    Selects the best Python interpreter.
    Prefers the local virtual environment (.venv) if available.
    """
    venv_python = ROOT_DIR / ".venv" / "Scripts" / "python.exe"
    if sys.platform != "win32":
        venv_python = ROOT_DIR / ".venv" / "bin" / "python"

    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Checks whether a TCP port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def wait_for_service(url: str, timeout: float = 35.0, label: str = "Service") -> bool:
    """
    Polls an HTTP endpoint until it returns a successful HTTP response (200, 302, 404).
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "HealthCheck/1.0"}
            )
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                if resp.status in (200, 302, 404):
                    return True
        except (urllib.error.HTTPError, urllib.error.URLError, socket.timeout, ConnectionRefusedError, OSError):
            pass
        time.sleep(0.5)
    return False


def get_database_status_display() -> str:
    """
    Determines active database connectivity (PostgreSQL vs SQLite fallback).
    """
    try:
        from src.config import DATABASE_URL, DB_PATH
        if DATABASE_URL:
            # Mask password in URL for secure console display
            import re
            masked_url = re.sub(r':([^@]+)@', ':****@', DATABASE_URL)
            target = masked_url.split("@")[-1] if "@" in masked_url else masked_url
            return f"🐘 PostgreSQL ({target})"
        return f"📁 SQLite Database ({DB_PATH.name})"
    except Exception:
        return "📁 SQLite Database (data/anomaly_events.db)"


def kill_process_tree(proc: subprocess.Popen):
    """
    Gracefully terminates a process and its child subprocess tree.
    Prevents orphaned Uvicorn/Streamlit background processes.
    """
    if proc is None or proc.poll() is not None:
        return

    try:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False
            )
        else:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Full-Stack Video Anomaly Detection Launcher")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind services (default: 127.0.0.1)")
    parser.add_argument("--port-api", type=int, default=8000, help="FastAPI backend port (default: 8000)")
    parser.add_argument("--port-ui", type=int, default=8501, help="Streamlit dashboard port (default: 8501)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the dashboard in browser automatically")
    parser.add_argument("--timeout", type=int, default=40, help="Maximum seconds to wait for service readiness (default: 40)")
    args = parser.parse_args()

    python_bin = find_python_executable()

    print("=" * 65)
    print("  🚀 Starting Video Anomaly Detection System (Full Stack)")
    print("=" * 65)
    print(f"Working Directory:   {ROOT_DIR}")
    print(f"Python Interpreter:  {python_bin}")

    # 1. Pre-flight Port Conflict Check
    if is_port_in_use(args.port_api, args.host):
        print(f"\n[⚠️ Warning] Port {args.port_api} is already in use.")
        print(f"  A previous FastAPI/Uvicorn instance may already be running.")
        print(f"  Please free port {args.port_api} or specify --port-api <port>.\n")

    if is_port_in_use(args.port_ui, args.host):
        print(f"\n[⚠️ Warning] Port {args.port_ui} is already in use.")
        print(f"  A previous Streamlit instance may already be running.")
        print(f"  Please free port {args.port_ui} or specify --port-ui <port>.\n")

    backend_url = f"http://{args.host}:{args.port_api}"
    frontend_url = f"http://localhost:{args.port_ui}"
    db_display = get_database_status_display()

    backend_proc = None
    frontend_proc = None

    try:
        # 2. Start Backend REST API (FastAPI)
        print(f"\n[1/2] Launching Backend API (FastAPI) at {backend_url} ...")
        backend_cmd = [
            python_bin, "-m", "uvicorn",
            "src.api.main:app",
            "--host", args.host,
            "--port", str(args.port_api)
        ]
        backend_proc = subprocess.Popen(
            backend_cmd,
            cwd=str(ROOT_DIR)
        )

        # 3. Start Frontend Web Dashboard (Streamlit)
        print(f"[2/2] Launching Frontend Dashboard (Streamlit) at {frontend_url} ...")
        frontend_cmd = [
            python_bin, "-m", "streamlit", "run",
            "src/dashboard/app.py",
            "--server.port", str(args.port_ui),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ]
        frontend_proc = subprocess.Popen(
            frontend_cmd,
            cwd=str(ROOT_DIR)
        )

        # 4. Active Health Verification
        print("\n⏳ Initializing neural network weights and waiting for services...")
        backend_ready = wait_for_service(
            f"{backend_url}/api/v1/health",
            timeout=args.timeout,
            label="Backend API"
        )
        frontend_ready = wait_for_service(
            frontend_url,
            timeout=args.timeout,
            label="Frontend Dashboard"
        )

        # Check if any process terminated prematurely
        if backend_proc.poll() is not None:
            print(f"\n[❌ Error] Backend API process terminated unexpectedly (code {backend_proc.returncode}).")
        if frontend_proc.poll() is not None:
            print(f"\n[❌ Error] Frontend Dashboard process terminated unexpectedly (code {frontend_proc.returncode}).")

        # 5. Full Stack System Banner
        print("\n" + "=" * 65)
        if backend_ready and frontend_ready:
            print("  ✅ FULL STACK SYSTEM IS ACTIVE & VERIFIED")
        else:
            print("  ⚠️ FULL STACK SYSTEM STARTED (Initializing)")
        print("=" * 65)
        print(f"  🌐 Frontend Dashboard: {frontend_url}")
        print(f"  ⚡ Backend REST API:   {backend_url}/docs")
        print(f"  📊 Database Status:    {db_display}")
        print("-" * 65)
        print("  Press Ctrl+C in this window to stop both Backend and Frontend.")
        print("=" * 65 + "\n")

        # 6. Auto-Open Browser
        if not args.no_browser and frontend_ready:
            try:
                webbrowser.open(frontend_url)
            except Exception:
                pass

        # 7. Keep Main Process Alive while Services Run
        while True:
            # If frontend exits, initiate shutdown
            if frontend_proc.poll() is not None:
                print("\n[*] Frontend Dashboard was closed.")
                break
            # If backend crashes, notify and initiate shutdown
            if backend_proc.poll() is not None:
                print("\n[!] Backend API exited unexpectedly.")
                break
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n[!] Stop signal received (Ctrl+C).")
    finally:
        print("[!] Gracefully terminating all background services...")
        if frontend_proc:
            kill_process_tree(frontend_proc)
        if backend_proc:
            kill_process_tree(backend_proc)
        print("  ✓ Backend REST API stopped.")
        print("  ✓ Frontend Dashboard stopped.")
        print("[*] All services stopped safely. Ports released.")


if __name__ == "__main__":
    main()

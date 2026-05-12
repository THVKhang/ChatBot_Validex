import socket
import subprocess
import os
import sys
import time
import signal
import atexit

backend_process = None
frontend_process = None

def is_port_in_use(port: int) -> bool:
    """Check if a port is currently in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

def kill_process_on_port(port: int):
    """Forcefully kill any process listening on the given port (Windows specific)."""
    try:
        result = subprocess.check_output(f"netstat -ano | findstr :{port}", shell=True, text=True)
        for line in result.strip().split('\n'):
            if "LISTENING" in line:
                parts = line.strip().split()
                if len(parts) >= 5:
                    pid = parts[-1]
                    if pid != "0":
                        subprocess.run(["taskkill", "/F", "/T", "/PID", pid], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass

def cleanup():
    """Forcefully terminate child processes when exiting."""
    print("\n🛑 Shutting down servers gracefully...")
    if os.name == "nt":
        if backend_process and backend_process.poll() is None:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(backend_process.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if frontend_process and frontend_process.poll() is None:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(frontend_process.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        if backend_process and backend_process.poll() is None:
            backend_process.terminate()
        if frontend_process and frontend_process.poll() is None:
            frontend_process.terminate()

atexit.register(cleanup)

def handle_signal(signum, frame):
    sys.exit(0)

# Catch termination signals
signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)
if hasattr(signal, "SIGBREAK"):
    signal.signal(signal.SIGBREAK, handle_signal)

def main():
    global backend_process, frontend_process

    print("🧹 Cleaning up old Zombie processes...")
    if os.name == "nt":
        kill_process_on_port(8000)
        kill_process_on_port(8001)
    
    api_port = 8000
    ui_port = 8001

    print("==================================================")
    print(f"🌟 Backend (API) will run on port : {api_port}")
    print(f"🌟 Frontend (UI) will run on port  : {ui_port}")
    print("==================================================")

    print("🚀 Starting Backend...")
    env = os.environ.copy()
    
    python_exe = sys.executable
    if sys.prefix == sys.base_prefix:
        venv_python = os.path.join(os.getcwd(), ".venv", "Scripts", "python.exe")
        if os.path.exists(venv_python):
            print(f"⚠️ Not in virtualenv. Auto-switching to: {venv_python}")
            python_exe = venv_python
            
    popen_kwargs = {}
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    backend_cmd = [python_exe, "-m", "uvicorn", "app.api_server:app", "--host", "0.0.0.0", "--port", str(api_port), "--reload"]
    backend_process = subprocess.Popen(backend_cmd, env=env, **popen_kwargs)

    time.sleep(2)

    print("🚀 Starting Frontend...")
    npm_cmd = "npm.cmd" if os.name == "nt" else "npm"
    frontend_cmd = [npm_cmd, "start", "--", "--port", str(ui_port)]
    frontend_dir = os.path.join(os.getcwd(), "ui", "angular-frontend")
    
    # Use shell=True for npm.cmd to prevent "Terminate batch job" interception hanging the console
    if os.name == "nt":
        frontend_process = subprocess.Popen(f"{npm_cmd} start -- --port {ui_port}", cwd=frontend_dir, shell=True)
    else:
        frontend_process = subprocess.Popen(frontend_cmd, cwd=frontend_dir)

    print("\n✅ All systems running!")
    print(f"👉 OPEN YOUR BROWSER AT: http://localhost:{ui_port}\n")
    print("Press Ctrl+C to stop both servers.\n")

    # Keep main thread alive and responsive to signals
    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()

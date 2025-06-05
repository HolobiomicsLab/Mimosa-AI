import os
import subprocess
import sys
import signal
from pathlib import Path

# Global list to keep track of running processes
processes = []

def find_api_files(root_dir):
    """Find all api.py files in subdirectories"""
    api_files = []
    for root, _, files in os.walk(root_dir):
        if 'api.py' in files:
            api_files.append(Path(root) / 'api.py')
    return api_files

def start_api_server(api_path, port):
    """Start an API server on specified port"""
    cmd = [sys.executable, str(api_path), str(port)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    processes.append(proc)
    return proc

def cleanup(signum, frame):
    """Cleanup all running processes on exit"""
    for proc in processes:
        proc.terminate()
    sys.exit(0)

def main():
    # Register signal handlers for clean exit
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    root_dir = os.path.dirname(os.path.abspath(__file__))
    api_files = find_api_files(root_dir)

    if not api_files:
        print("No api.py files found in subdirectories")
        return

    print(f"Found {len(api_files)} API servers to start:")
    for i, api_file in enumerate(api_files):
        port = 5000 + i
        print(f"Starting {api_file} on port {port}")
        start_api_server(api_file, port)

    print("\nAll APIs running. Press Ctrl+C to stop.")
    signal.pause()  # Wait indefinitely until interrupted

if __name__ == "__main__":
    main()

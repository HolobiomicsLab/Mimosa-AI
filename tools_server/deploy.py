import os
import subprocess
import sys
import signal
from pathlib import Path
import select

# Global list to keep track of running processes and their info
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
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    processes.append({'proc': proc, 'api_path': api_path, 'port': port})
    return proc

def cleanup(_signum, _frame):
    """Cleanup all running processes on exit"""
    for p in processes:
        p['proc'].terminate()
    sys.exit(0)

def monitor_processes():
    """Monitor all running processes and print their stdout to stdout"""
    import time
    while processes:
        for p in processes[:]:  # Create a copy to safely modify during iteration
            proc = p['proc']
            # Read available output without blocking
            try:
                while True:
                    line = proc.stdout.readline()
                    if not line:
                        break
                    print(f"[{p['api_path']}:{p['port']}] {line}", end='')
            except:
                pass
            
            # Check if process has exited
            if proc.poll() is not None:
                print(f"\nProcess {p['api_path']} on port {p['port']} exited.")
                processes.remove(p)
        
        time.sleep(0.1)  # Small delay to prevent high CPU usage
    
    print("All processes exited.")

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

    print("\nAll APIs running. Press Ctrl+C to stop.\n")
    monitor_processes()

if __name__ == "__main__":
    main()

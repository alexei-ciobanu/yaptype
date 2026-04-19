"""
yaptype — push-to-talk voice dictation manager.

Usage:
    uv run --no-sync python yapctl.py start        # Start server + dictation client
    uv run --no-sync python yapctl.py stop         # Stop the background server
    uv run --no-sync python yapctl.py status       # Check server status
    uv run --no-sync python yapctl.py restart      # Restart server + dictation client
    uv run --no-sync python yapctl.py server       # Start ONLY the server (background)
    uv run --no-sync python yapctl.py logs [-f]    # View server logs
"""

import argparse
import os
import signal
import socket
import subprocess
import sys
import time

# PID file location
PID_DIR = os.path.join(os.path.expanduser("~"), ".yaptype")
PID_FILE = os.path.join(PID_DIR, "server.pid")
LOG_FILE = os.path.join(PID_DIR, "server.log")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9876


def is_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    """Check if a TCP port is accepting connections."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect((host, port))
        sock.close()
        return True
    except (ConnectionRefusedError, TimeoutError, OSError):
        return False


def is_process_alive(pid: int) -> bool:
    """Check if a process with the given PID is still running."""
    if sys.platform == "win32":
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True,
                text=True,
            )
            return str(pid) in result.stdout
        except OSError:
            return False
    else:
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False


def read_pid() -> int | None:
    """Read PID from file, return None if not found or stale."""
    if not os.path.exists(PID_FILE):
        return None
    try:
        with open(PID_FILE) as f:
            pid = int(f.read().strip())
        if is_process_alive(pid):
            return pid
        os.remove(PID_FILE)
        return None
    except (ValueError, OSError):
        return None


def write_pid(pid: int):
    """Write PID to file."""
    os.makedirs(PID_DIR, exist_ok=True)
    with open(PID_FILE, "w") as f:
        f.write(str(pid))


def remove_pid():
    """Remove PID file."""
    if os.path.exists(PID_FILE):
        os.remove(PID_FILE)


def ensure_server(args) -> bool:
    """Ensure the ASR server is running. Returns True if ready."""
    existing_pid = read_pid()
    if existing_pid is not None and is_port_open(args.host, args.port):
        print(f"Server already running (PID {existing_pid})")
        return True

    if existing_pid is not None:
        print(f"Stale server process (PID {existing_pid}), cleaning up...")
        cmd_stop(args)

    if is_port_open(args.host, args.port):
        print(f"Port {args.port} is already in use by another process!")
        return False

    # Build the server command
    script_dir = os.path.dirname(os.path.abspath(__file__))
    backend = getattr(args, "backend", "nemo")
    if backend == "llama":
        server_script = os.path.join(script_dir, "asr_server_llama.py")
    else:
        server_script = os.path.join(script_dir, "asr_server.py")
    venv_python = os.path.join(script_dir, ".venv", "Scripts", "python.exe")

    if not os.path.exists(venv_python):
        venv_python = sys.executable

    cmd = [venv_python, server_script, "--host", args.host, "--port", str(args.port)]

    # Start process in background (no terminal window on Windows)
    os.makedirs(PID_DIR, exist_ok=True)
    log_fh = open(LOG_FILE, "w")  # noqa: SIM115  -- kept open for subprocess lifetime

    creation_flags = 0
    if sys.platform == "win32":
        creation_flags = subprocess.CREATE_NO_WINDOW

    proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        creationflags=creation_flags,
    )

    write_pid(proc.pid)
    print(f"Starting server (PID {proc.pid})...")

    # Wait for server to become ready
    print("Waiting for model to load", end="", flush=True)
    for _ in range(120):
        time.sleep(1)
        print(".", end="", flush=True)

        if proc.poll() is not None:
            print(f"\nServer failed to start! Exit code: {proc.returncode}")
            print(f"Check logs: {LOG_FILE}")
            remove_pid()
            return False

        if is_port_open(args.host, args.port):
            print(f"\nServer ready! (PID {proc.pid}, {args.host}:{args.port})")
            return True

    print(f"\nTimeout waiting for server. Check logs: {LOG_FILE}")
    return False


def run_client(args):
    """Run the dictation client in the foreground."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    client_script = os.path.join(script_dir, "dictate_client.py")
    venv_python = os.path.join(script_dir, ".venv", "Scripts", "python.exe")

    if not os.path.exists(venv_python):
        venv_python = sys.executable

    cmd = [
        venv_python,
        client_script,
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    if hasattr(args, "key") and args.key:
        cmd.extend(["--key", args.key])
    if hasattr(args, "device") and args.device is not None:
        cmd.extend(["--device", str(args.device)])

    import contextlib

    with contextlib.suppress(KeyboardInterrupt):
        subprocess.run(cmd)


def cmd_start(args):
    """Start server (if needed) and run dictation client."""
    if not ensure_server(args):
        return
    print()
    run_client(args)


def cmd_server(args):
    """Start only the server in background."""
    ensure_server(args)


def cmd_stop(args):
    """Stop the ASR server."""
    pid = read_pid()
    if pid is None:
        print("Server is not running.")
        return

    print(f"Stopping server (PID {pid})...", end="", flush=True)

    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)
        else:
            os.kill(pid, signal.SIGTERM)

        for _ in range(10):
            if not is_process_alive(pid):
                break
            time.sleep(0.5)

        if is_process_alive(pid):
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)
            else:
                os.kill(pid, signal.SIGKILL)  # type: ignore[attr-defined]  # Unix only
            time.sleep(0.5)

        print(" done")
    except (OSError, ProcessLookupError):
        print(" (already stopped)")

    remove_pid()


def cmd_status(args):
    """Check server status."""
    pid = read_pid()
    port_open = is_port_open(args.host, args.port)

    if pid is not None and port_open:
        print(f"Server is RUNNING (PID {pid}, {args.host}:{args.port})")
    elif pid is not None and not port_open:
        print(f"Server process exists (PID {pid}) but port {args.port} is not responding")
        print("  The server may still be loading the model...")
    elif pid is None and port_open:
        print(f"Port {args.port} is open but not managed by yapctl")
        print("  (may be a manually started server)")
    else:
        print("Server is NOT running.")


def cmd_restart(args):
    """Restart server and run dictation client."""
    cmd_stop(args)
    time.sleep(1)
    cmd_start(args)


def cmd_logs(args):
    """Show server logs."""
    if not os.path.exists(LOG_FILE):
        print("No log file found.")
        return

    with open(LOG_FILE) as f:
        content = f.read()

    if args.follow:
        lines = content.splitlines()
        for line in lines[-20:]:
            print(line)
        try:
            with open(LOG_FILE) as f:
                f.seek(0, 2)
                while True:
                    line = f.readline()
                    if line:
                        print(line, end="")
                    else:
                        time.sleep(0.5)
        except KeyboardInterrupt:
            pass
    else:
        print(content)


def main():
    parser = argparse.ArgumentParser(
        description="yaptype — push-to-talk voice dictation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  %(prog)s start                   Start server + dictation (Ctrl+C to stop dictating)
  %(prog)s start --key scroll_lock Use Scroll Lock as hotkey
  %(prog)s stop                    Stop the background server
  %(prog)s status                  Check if server is running
  %(prog)s server                  Start only the server (background)
  %(prog)s logs -f                 Follow server logs""",
    )
    parser.add_argument(
        "--host", default=DEFAULT_HOST, help=f"Server host (default: {DEFAULT_HOST})"
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help=f"Server port (default: {DEFAULT_PORT})"
    )
    parser.add_argument(
        "--backend",
        choices=["nemo", "llama"],
        default="nemo",
        help="ASR backend: nemo (Parakeet) or llama (Qwen3-ASR via llama-server) (default: nemo)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    start_parser = subparsers.add_parser("start", help="Start server + run dictation client")
    start_parser.add_argument(
        "--key",
        default="ctrl_r",
        help="Hotkey to hold for recording (default: ctrl_r)",
    )
    start_parser.add_argument("--device", type=int, default=None, help="Audio input device index")

    subparsers.add_parser("server", help="Start only the server in background")
    subparsers.add_parser("stop", help="Stop the background server")
    subparsers.add_parser("status", help="Check server status")

    restart_parser = subparsers.add_parser("restart", help="Restart server + run dictation client")
    restart_parser.add_argument(
        "--key",
        default="ctrl_r",
        help="Hotkey to hold for recording (default: ctrl_r)",
    )
    restart_parser.add_argument("--device", type=int, default=None, help="Audio input device index")

    logs_parser = subparsers.add_parser("logs", help="Show server logs")
    logs_parser.add_argument("-f", "--follow", action="store_true", help="Follow log output")

    args = parser.parse_args()

    commands = {
        "start": cmd_start,
        "server": cmd_server,
        "stop": cmd_stop,
        "status": cmd_status,
        "restart": cmd_restart,
        "logs": cmd_logs,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()

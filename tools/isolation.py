"""
Execution Isolation & Process Boundary Engine for CyberJarvis.

Provides explicit execution-isolation policies:
- PROCESS: For mutating operations, external processes (pytest, git), network operations,
  and tools requiring killable process boundaries to guarantee that delayed operations
  or runaway threads do NOT continue executing after a timeout.
- THREAD: For lightweight, short-running, CPU/in-memory or internally bounded tools
  (e.g., datetime, kernel stats via psutil) where execution latency is negligible.

Timeout Semantics & Caveats:
- Process isolation enforces a hard killable boundary using process groups (os.setpgid)
  and recursive child cleanup (psutil). On timeout, SIGTERM followed by SIGKILL is sent
  to the entire process group.
- Killing a process stops execution immediately and prevents future delayed operations.
  It does NOT roll back side effects (such as filesystem writes or network requests)
  that already completed before the termination signal was received.
"""
import os
import sys
import time
import signal
from enum import Enum
from typing import Callable, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import multiprocessing as mp

from observability.logging import get_logger

logger = get_logger("CyberJarvis.Isolation")


class IsolationMode(str, Enum):
    """Execution isolation mode for registered tools."""
    THREAD = "thread"
    PROCESS = "process"


def _get_mp_context() -> mp.context.BaseContext:
    """
    Select multiprocessing start method context.
    Uses 'fork' on Linux for zero-overhead execution of registered tool callables
    (including closures in test harnesses without pickling restrictions)
    and clean POSIX process group separation.
    """
    method = os.environ.get("CYBERJARVIS_MP_START_METHOD", "fork")
    try:
        return mp.get_context(method)
    except ValueError:
        return mp.get_context("fork")


def terminate_process_tree(pid: int, grace_period_sec: float = 0.05) -> None:
    """
    Terminates a process and all its child/grandchild processes on Linux.
    Uses POSIX process group signaling (os.killpg) and psutil recursive tree
    termination as defense-in-depth.
    """
    # 1. Enumerate children using psutil before signals if possible
    children = []
    try:
        import psutil
        try:
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            children = []
    except ImportError:
        children = []

    # 2. Get process group ID
    pgid = None
    try:
        pgid = os.getpgid(pid)
    except (ProcessLookupError, OSError):
        pass

    # 3. SIGTERM to process group
    if pgid is not None:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass
    else:
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass

    for c in children:
        try:
            c.terminate()
        except Exception:
            pass

    if grace_period_sec > 0:
        time.sleep(grace_period_sec)

    # 4. SIGKILL to process group to guarantee no survivors
    if pgid is not None:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass
    else:
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass

    for c in children:
        try:
            c.kill()
        except Exception:
            pass


def _process_worker_entrypoint(func: Callable, kwargs: Dict[str, Any], send_conn) -> None:
    """
    Worker process entry point.
    Runs in a dedicated process group (os.setpgid) so all child processes
    inherit the group and can be terminated together on timeout.
    """
    try:
        os.setpgid(0, 0)
    except Exception:
        pass

    try:
        output = func(**kwargs)
        try:
            send_conn.send({"status": "ok", "result": output})
        except Exception as send_err:
            send_conn.send({
                "status": "error",
                "error_type": "SerializationError",
                "error_message": f"Tool output could not be serialized: {send_err}",
            })
    except Exception as ex:
        try:
            send_conn.send({
                "status": "error",
                "error_type": type(ex).__name__,
                "error_message": str(ex),
            })
        except Exception:
            pass
    finally:
        try:
            send_conn.close()
        except Exception:
            pass
        # Immediate exit without executing parent stream flush or atexit handlers
        os._exit(0)


def execute_in_process(
    func: Callable,
    kwargs: Dict[str, Any],
    timeout_sec: float,
) -> Tuple[bool, Any, Optional[str], float]:
    """
    Execute a tool function inside an isolated, killable process boundary.

    Returns:
        (success, result, error_message, duration_ms)
    """
    ctx = _get_mp_context()
    recv_conn, send_conn = ctx.Pipe(duplex=False)

    sys.stdout.flush()
    sys.stderr.flush()

    start_time = time.perf_counter()
    p = ctx.Process(target=_process_worker_entrypoint, args=(func, kwargs, send_conn))
    p.start()
    send_conn.close()

    try:
        has_data = recv_conn.poll(timeout_sec)
    except Exception as ex:
        has_data = False

    duration_ms = (time.perf_counter() - start_time) * 1000

    if not has_data:
        # Hard timeout enforcement: terminate worker process and its children
        terminate_process_tree(p.pid)
        p.join(timeout=0.5)
        if p.is_alive():
            try:
                os.kill(p.pid, signal.SIGKILL)
            except OSError:
                pass
            p.join(timeout=0.2)
        try:
            recv_conn.close()
        except Exception:
            pass
        err_msg = f"Tool execution timed out after {timeout_sec}s."
        return False, None, err_msg, duration_ms

    # Worker finished within timeout
    try:
        payload = recv_conn.recv()
    except EOFError:
        payload = {
            "status": "error",
            "error_type": "ProcessDied",
            "error_message": "Worker process exited prematurely without returning data.",
        }
    except Exception as ex:
        payload = {
            "status": "error",
            "error_type": "IPCError",
            "error_message": f"Failed to receive data from worker process: {ex}",
        }
    finally:
        try:
            recv_conn.close()
        except Exception:
            pass
        p.join(timeout=0.5)

    if payload.get("status") == "ok":
        return True, payload.get("result"), None, duration_ms
    else:
        err_type = payload.get("error_type", "ExecutionError")
        err_msg = payload.get("error_message", "Unknown worker error")
        return False, None, f"{err_msg}", duration_ms


def execute_in_thread(
    func: Callable,
    kwargs: Dict[str, Any],
    timeout_sec: float,
) -> Tuple[bool, Any, Optional[str], float]:
    """
    Execute a tool function inside a thread pool.
    Suitable ONLY for short-running, CPU/in-memory or internally bounded tools.
    On timeout, shuts down pool without waiting to return immediately to caller.
    """
    start_time = time.perf_counter()
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(func, **kwargs)

    try:
        output = future.result(timeout=timeout_sec)
        duration_ms = (time.perf_counter() - start_time) * 1000
        pool.shutdown(wait=False)
        return True, output, None, duration_ms
    except TimeoutError:
        duration_ms = (time.perf_counter() - start_time) * 1000
        pool.shutdown(wait=False, cancel_futures=True)
        err_msg = f"Tool execution timed out after {timeout_sec}s."
        return False, None, err_msg, duration_ms
    except Exception as ex:
        duration_ms = (time.perf_counter() - start_time) * 1000
        pool.shutdown(wait=False)
        return False, None, str(ex), duration_ms

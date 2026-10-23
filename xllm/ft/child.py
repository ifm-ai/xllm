"""
Starts one trainer under the ft agent: `python -m xllm.ft.child train.py <args>` makes this process die with its
agent (PR_SET_PDEATHSIG, kept across exec), then runs `python train.py <args>` in it, so train.py itself needs
nothing for it.
"""

import ctypes
import os
import signal
import sys

if __name__ == "__main__":
    ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG (the process's libc: glibc or musl)
    agent = os.environ.get("FT_AGENT_PID")
    if agent is None:
        print("xllm.ft.child: FT_AGENT_PID is not set (not started by the ft agent); no parent check", file=sys.stderr)
    elif os.getppid() != int(agent):  # the agent died before prctl: no signal will come
        os._exit(1)
    unbuffered = ["-u"] if sys.stdout.write_through else []  # as the agent started this process
    os.execv(sys.executable, [sys.executable] + unbuffered + sys.argv[1:])

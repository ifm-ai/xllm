import os
import contextlib
import faulthandler
import io
from typing import Optional
import platform
import signal
import tempfile


def execute(script, result, timeout):
    with create_tempdir():
        # These system calls are needed when cleaning up tempdir.
        import os
        import shutil

        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir

        # Disable functionalities that can make destructive changes to the test.
        reliability_guard()

        # Construct the check program and run it.
        check_program = script

        try:
            exec_globals = {}
            with swallow_io():
                with time_limit(timeout):
                    exec(check_program, exec_globals)
            result.append("passed")
        except TimeoutException:
            result.append("timed out")
        except BaseException as e:
            result.append(f"failed: {e}")
        # Needed for cleaning up.
        shutil.rmtree = rmtree
        os.rmdir = rmdir
        os.chdir = chdir


@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield


@contextlib.contextmanager
def create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname


class TimeoutException(Exception):
    pass


class WriteOnlyStringIO(io.StringIO):
    """StringIO that throws an exception when it's read from"""

    def read(self, *args, **kwargs):
        raise IOError

    def readline(self, *args, **kwargs):
        raise IOError

    def readlines(self, *args, **kwargs):
        raise IOError

    def readable(self, *args, **kwargs):
        """Returns True if the IO object can be read."""
        return False


class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = "stdin"


@contextlib.contextmanager
def chdir(root):
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    except BaseException as exc:
        raise exc
    finally:
        os.chdir(cwd)


def reliability_guard(maximum_memory_bytes: Optional[int] = None):
    """
    This disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)
    WARNING
    This function is NOT a security sandbox. Untrusted code, including, model-
    generated code, should not be blindly executed outside of one. See the
    Codex paper for more information about OpenAI's code sandbox, and proceed
    with caution.
    """

    if maximum_memory_bytes is not None:
        import resource

        resource.setrlimit(
            resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes)
        )
        resource.setrlimit(
            resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes)
        )
        if not platform.uname().system == "Darwin":
            resource.setrlimit(
                resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes)
            )

    faulthandler.disable()

    import builtins

    builtins.exit = None  # type: ignore
    builtins.quit = None  # type: ignore

    import os

    os.environ["OMP_NUM_THREADS"] = "1"

    os.kill = None  # type: ignore
    os.system = None  # type: ignore
    os.putenv = None  # type: ignore
    os.remove = None  # type: ignore
    os.removedirs = None  # type: ignore
    os.rmdir = None  # type: ignore
    os.fchdir = None  # type: ignore
    os.setuid = None  # type: ignore
    os.fork = None  # type: ignore
    os.forkpty = None  # type: ignore
    os.killpg = None  # type: ignore
    os.rename = None  # type: ignore
    os.renames = None  # type: ignore
    os.truncate = None  # type: ignore
    os.replace = None  # type: ignore
    os.unlink = None  # type: ignore
    os.fchmod = None  # type: ignore
    os.fchown = None  # type: ignore
    os.chmod = None  # type: ignore
    os.chown = None  # type: ignore
    os.chroot = None  # type: ignore
    os.fchdir = None  # type: ignore
    os.lchflags = None  # type: ignore
    os.lchmod = None  # type: ignore
    os.lchown = None  # type: ignore
    os.getcwd = None  # type: ignore
    os.chdir = None  # type: ignore

    import shutil

    shutil.rmtree = None  # type: ignore
    shutil.move = None  # type: ignore
    shutil.chown = None  # type: ignore

    import subprocess

    subprocess.Popen = None  # type: ignore

    __builtins__["help"] = None  # type: ignore

    import sys

    sys.modules["ipdb"] = None  # type: ignore
    sys.modules["joblib"] = None  # type: ignore
    sys.modules["resource"] = None  # type: ignore
    sys.modules["psutil"] = None  # type: ignore
    sys.modules["tkinter"] = None  # type: ignore

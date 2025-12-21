import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional


@dataclass(frozen=True)
class RuntimeEnv:
    """
    Minimal runtime environment description applied around task exceution.

    Attributes:
        env_vars (Dict[str, str]): Environment variables to set.
        cwd (Optional[str]): Working directory to switch to.
        python_path (Optional[List[str]]): Additional paths to add to PYTHONPATH.

    Examples:
        >>> env = RuntimeEnv(env_vars={"MY_VAR": "value"}, cwd="/tmp", python_path=["/my/lib"])
    """

    env_vars: Optional[Dict[str, str]] = None
    cwd: Optional[str] = None
    python_path: Optional[List[str]] = None

    @contextmanager
    def apply(self) -> Iterator[None]:
        """
        Context manager that temporarily applies this runtime environment.

        Returns:
            Iterator[None]: Use as `with env.apply(): run_task()`.

        Notes:
            - We snapshot only what we change, then restore.
            - `PYTHONPATH` is prepended (if provided).
        """
        # snapshots
        old_environ = os.environ.copy()
        old_cwd = os.getcwd()
        old_python_path = os.environ.get("PYTHONPATH", "")

        try:
            # env vars
            if self.env_vars:
                os.environ.update({k: str(v) for k, v in self.env_vars.items()})

            # python path
            if self.python_path:
                prefix = os.pathsep.join(self.python_path)
                os.environ["PYTHONPATH"] = prefix + (os.pathsep + old_python_path if old_python_path else "")

            # cwd
            if self.cwd:
                os.chdir(self.cwd)
            yield

        finally:
            # restore everything we touched
            os.environ.clear()
            os.environ.update(old_environ)
            if self.cwd:
                os.chdir(old_cwd)
            # ensure PYTHONPATH mirrors snapshot (covered by environ restore but explicit)
            if old_python_path:
                os.environ["PYTHONPATH"] = old_python_path
            else:
                os.environ.pop("PYTHONPATH", None)

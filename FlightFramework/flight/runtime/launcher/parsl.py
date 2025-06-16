import typing as t
from concurrent.futures import Future

from flight.jobs import Job
from flight.runtime.launcher.base import Launcher
import time

class ParslLauncher(Launcher):
    """
    Class that launches tasks via [Parsl](https://parsl.readthedocs.io/en/stable/).
    """

    def __init__(
        self, config: dict[str, t.Any], run_dir: str = ".parsl", stript_dir=".parsl"
    ):
        super().__init__()

        import parsl
        from parsl.executors import HighThroughputExecutor

        parsl.load()

        self._config = config
        self.executor = HighThroughputExecutor(**self._config)
        self.executor.run_dir = run_dir
        self.executor.provider.script_dir = stript_dir
        self.executor.start()

        # Run priming job to reduce initial startup costs.
        fut = self.executor.submit(_platform_info, {})
        fut.result()
        print(f"priming_done:{time.perf_counter()}")

    def submit(self, job: Job, /, **kwargs) -> Future:
        future = self.executor.submit(job, {}, **kwargs)
        return future

    def collect(self):
        raise NotImplementedError


def _platform_info():
    import platform

    return platform.platform()

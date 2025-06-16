import typing as t
from concurrent.futures import Future

from flight.jobs import Job
from flight.runtime.launcher import Launcher
from flight.runtime.transfer import BaseTransfer

Config = t.NewType("Config", dict[str, t.Any])


class Borg:
    _shared_state: dict[str, t.Any] = {}

    def __init__(self):
        self.__dict__ = self._shared_state


class Runtime(Borg):
    launcher: Launcher
    transfer: BaseTransfer

    def __init__(self, launcher: Launcher, transfer: BaseTransfer):
        Borg.__init__(self)
        self.launcher = launcher
        self.transfer = transfer

    # TODO: Come up with typing for `Job = NewType("Job", Callable[[...], ...])`
    def submit(self, job: Job, /, **kwargs) -> Future:
        return self.launcher.submit(job, **kwargs, transfer=self.transfer)

    def proxy(self, data: t.Any):
        return self.transfer.proxy(data)

    # @classmethod
    # def create(
    #     cls, launcher_cfg: Config | None, transfer_cfg: Config | None
    # ) -> "Runtime":
    #     launcher_cfg = {} if launcher_cfg is None else launcher_cfg
    #     transfer_cfg = {} if transfer_cfg is None else transfer_cfg
    #     launcher = Launcher.create(**launcher_cfg)
    #     transfer = Transfer.create(**transfer_cfg)
    #     return Runtime(launcher, transfer)

"""Track owned descendants, including children that create separate sessions."""

import psutil


class OwnedProcessTree:
    """Track process identities while a workflow runs; never use bare PID reuse."""

    def __init__(self, pid):
        try:
            self.root = psutil.Process(pid)
        except psutil.NoSuchProcess:
            self.root = None
        self.children = {}

    def capture(self):
        """Remember descendants before a parent can exit and reparent them."""
        for parent in [self.root, *list(self.children.values())]:
            if parent is None:
                continue
            try:
                for child in parent.children(recursive=True):
                    try:
                        self.children[(child.pid, child.create_time())] = child
                    except psutil.NoSuchProcess:
                        continue
            except psutil.NoSuchProcess:
                continue

    def kill(self, include_root=True):
        """Stop spawning, kill known descendants and optionally the owned root."""
        if include_root:
            self._signal(self.root, "suspend")
        self.capture()
        for child in list(self.children.values()):
            self._signal(child, "suspend")
        self.capture()
        for child in reversed(list(self.children.values())):
            self._signal(child, "kill")
        if include_root:
            self._signal(self.root, "kill")

    @staticmethod
    def _signal(process, method):
        try:
            if process is not None and process.is_running():
                getattr(process, method)()
        except psutil.NoSuchProcess:
            pass

    def wait_children(self):
        """Wait for tracked children and fail if a live one survives cleanup."""
        _, alive = psutil.wait_procs(list(self.children.values()), timeout=2)
        survivors = []
        for process in alive:
            try:
                if process.status() != psutil.STATUS_ZOMBIE:
                    survivors.append(process.pid)
            except psutil.NoSuchProcess:
                continue
        if survivors:
            raise RuntimeError(f"Owned workflow descendants survived cleanup: {survivors}")

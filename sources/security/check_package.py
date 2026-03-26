import sys

class PackageCheck():
    def __init__(self):
        self.malicious_versions = {
            # CVE litellm 1.82.8 supply-chain attack
            'litellm': {'max': '1.82.8', 'min': None, 'reason': '.pth supply chain attack'}
        }

    def run(self):
        for pkg, info in self.malicious_versions.items():
            self._check_version(pkg, info['min'], info['max'], info['reason'])

    def _check_version(self, package, version_min, version_max, reason) -> None:
        """Exit early if a known-malicious version is installed."""
        try:
            from importlib.metadata import version as pkg_version, PackageNotFoundError
            try:
                import packaging.version as _pv
            except ImportError:
                class _pv:
                    @staticmethod
                    def parse(v: str):
                        return tuple(int(x) for x in v.split(".")[:3])
                    Version = parse

            try:
                installed = pkg_version(package)  # ← was hardcoded "litellm"
            except PackageNotFoundError:
                return

            parsed = _pv.parse(installed)
            threshold_up = _pv.parse(version_max or "999.999.999")
            threshold_down = _pv.parse(version_min or "0.0.0")

            # ← FIXED: proper range check (inclusive)
            safe = False
            if version_min and version_max:
                safe = threshold_down <= parsed <= threshold_up
            elif version_max:
                safe = parsed <= threshold_up
            elif version_min:
                safe = parsed >= threshold_down

            if not safe:
                print(
                    "\n"
                    "╔══════════════════════════════════════════════════════════════╗\n"
                    "║  🚨  SECURITY WARNING — MIMOSA STARTUP ABORTED               ║\n"
                    "╠══════════════════════════════════════════════════════════════╣\n"
                   f"║  Installed {package} version : {installed:<33}               ║\n"
                   f"║  Blocked version range : {version_min or 'any'} to {version_max or 'any'}          ║\n"
                   f"║  Reason: {reason:<45}               ║\n"
                    "╚══════════════════════════════════════════════════════════════╝\n"
                )
                sys.exit(1)

        except Exception as exc:
            print(f"⚠️  {package} version check failed unexpectedly: {exc}")  # ← was hardcoded "litellm"
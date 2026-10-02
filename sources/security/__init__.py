"""
Security sub-package for Mimosa-AI.

Provides dependency-version checks and other security-related utilities
to verify the runtime environment before execution.
"""

from .check_package import PackageCheck

__all__ = [
    "PackageCheck",
]

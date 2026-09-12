"""CLI interfaces loaded on demand, without initializing unrelated workflows."""

from importlib import import_module

_EXPORT_MODULES = {
    'OnboardCLI': 'onboard_cli',
    'EvaluationCLI': 'evaluation_cli',
    'MemoryChatCLI': 'memory_chat_cli',
    'print_ok': 'pretty_print',
    'print_warn': 'pretty_print',
    'print_err': 'pretty_print',
    'print_info': 'pretty_print',
    'print_step': 'pretty_print',
    'print_phase': 'pretty_print',
    'print_section': 'pretty_print',
    'print_rule': 'pretty_print',
    'print_iteration_header': 'pretty_print',
    'print_box': 'pretty_print',
    'print_kv_row': 'pretty_print',
    'print_summary': 'pretty_print',
    'print_agent_answers': 'pretty_print',
    'CYAN': 'pretty_print',
    'GREEN': 'pretty_print',
    'YELLOW': 'pretty_print',
    'RED': 'pretty_print',
    'BLUE': 'pretty_print',
    'MAGENTA': 'pretty_print',
    'BOLD': 'pretty_print',
    'DIM': 'pretty_print',
    'RESET': 'pretty_print',
}

__all__ = [
    "OnboardCLI",
    "EvaluationCLI",
    "MemoryChatCLI",
    "print_ok",
    "print_warn",
    "print_err",
    "print_info",
    "print_step",
    "print_phase",
    "print_section",
    "print_rule",
    "print_iteration_header",
    "print_box",
    "print_kv_row",
    "print_summary",
    "print_agent_answers",
    "CYAN", "GREEN", "YELLOW", "RED", "BLUE", "MAGENTA", "BOLD", "DIM", "RESET",
]


def __getattr__(name):
    """Resolve and cache an existing CLI export from its owning module."""
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{module_name}", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    """List CLI exports without initializing interactive execution modes."""
    return sorted(set(globals()) | set(__all__))

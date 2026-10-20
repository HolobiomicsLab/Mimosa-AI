"""Frozen comparison policies and schema adapters in a restrictive worker.

Generated source is interpreted, never passed to exec or eval. The worker exposes
only JSON values, pure operations and confined artifact reads. AST screening is
an input check, not an OS sandbox. This deliberately limited language rejects
unsupported scientific formats instead of inventing measurements.
"""

from __future__ import annotations

import ast
import copy
import csv
import hashlib
import io
import json
import math
import operator
import os
import resource
import signal
import struct
import subprocess
import sys
import tempfile
from collections import ChainMap
from pathlib import Path

SOURCE_BYTES = 131072
WORKER_BYTES = 16 * 1024 * 1024
READ_BYTES = 64 * 1024 * 1024
MEMORY_BYTES = 1024 * 1024 * 1024
# Transcript keys the guard writes; copied into the frozen bundle.
GUARD_RECORD_KEYS = ("criteria_guard", "plan_revisions", "dropped_criteria")


class PolicyError(ValueError):
    """The comparison instrument failed; this is not a task loss.

    Callers: the repair loop and the evaluator's error boundary. Any PolicyError
    from a comparison is a verifier-side fault, never evidence against a workflow.
    It is retried through LLM repair rounds and, after the last round, marks the
    generation ``verifier_instrument_error`` (no reward) instead of scoring it.
    """


class AdaptationError(PolicyError):
    """A schema could not be normalized under the frozen task contract.

    Callers: adapter execution inside the worker. Raised when generated adapter
    code cannot read or parse an artifact schema. The repair loop routes it to
    adapter repair (parsing fix), never to comparison repair.
    """


class _Return(Exception):
    """Carry a return value through interpreted control flow.

    Used only by Interpreter.apply to unwind a generated function's return
    statement; never escapes the interpreter.
    """


class _Break(Exception):
    """Stop an interpreted loop.

    Used only by Interpreter.loop to unwind a generated break statement.
    """


class _Continue(Exception):
    """Advance an interpreted loop.

    Used only by Interpreter.loop to unwind a generated continue statement.
    """


BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
          ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
          ast.Mod: operator.mod, ast.Pow: operator.pow, ast.BitAnd: operator.and_,
          ast.BitOr: operator.or_}
COMPARE = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
           ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge,
           ast.Is: operator.is_, ast.IsNot: operator.is_not,
           ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b}
BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
    "enumerate": enumerate, "float": float, "int": int, "isinstance": isinstance,
    "len": len, "list": list, "max": max, "min": min, "range": range,
    "reversed": reversed, "round": round, "set": set, "sorted": sorted, "str": str,
    "sum": sum, "tuple": tuple, "zip": zip, "ValueError": ValueError,
}
#: Builtins a program may read as values (isinstance targets, None, etc.).
VALUE_NAMES = {**{name: BUILTINS[name] for name in
                  ("bool", "dict", "float", "int", "list", "set", "str", "tuple")},
               "ValueError": ValueError}
MODULE_CALLS = {"json": {"loads": json.loads, "dumps": json.dumps},
                "math": {name: getattr(math, name) for name in
                         ("isfinite", "sqrt", "log", "exp", "fabs", "isclose")},
                "struct": {"unpack": struct.unpack, "calcsize": struct.calcsize}}
METHODS = {
    str: {"split", "splitlines", "strip", "lstrip", "rstrip", "lower", "upper",
          "replace", "startswith", "endswith", "join", "find", "count",
          "isdigit", "isnumeric", "removeprefix", "removesuffix"},
    bytes: {"decode", "split", "strip", "startswith", "endswith", "find"},
    dict: {"get", "items", "keys", "values", "update", "setdefault"},
    list: {"append", "extend", "count", "index"}, set: {"add"},
}
CONVERSIONS = {-1: lambda value: value, ord("s"): str, ord("r"): repr, ord("a"): ascii}
ALLOWED_NODES = {
    ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Assign, ast.AugAssign,
    ast.Expr, ast.If, ast.For, ast.Return, ast.Break, ast.Continue, ast.Raise,
    ast.Pass, ast.Name, ast.Constant, ast.List, ast.Tuple, ast.Set, ast.Dict,
    ast.Subscript, ast.Slice, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare,
    ast.IfExp, ast.Call, ast.Attribute, ast.keyword, ast.ListComp, ast.DictComp,
    ast.SetComp, ast.GeneratorExp, ast.comprehension, ast.Load, ast.Store,
    ast.JoinedStr, ast.FormattedValue,
    ast.And, ast.Or, ast.Not, ast.USub, ast.UAdd, *BINARY, *COMPARE,
}


def screen_source(source: str) -> ast.Module:
    """Check the input language before the restrictive interpreter sees it.

    Called by PolicyEngine.source on every LLM answer, on untrusted generated
    Python. Rejects anything outside the allowed statement and expression set:
    no imports, while, try, lambda, class, decorators, variadic parameters,
    async, private or dunder names, and sources over SOURCE_BYTES.

    Args:
        source (str): Untrusted generated Python source.

    Returns:
        ast.Module: Parsed program, safe for Interpreter.

    Raises:
        PolicyError: For unsupported syntax, private names, or oversized source.
    """
    if not isinstance(source, str) or len(source.encode()) > SOURCE_BYTES:
        raise PolicyError("Policy source is missing or too large")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if type(node) not in ALLOWED_NODES:
            raise PolicyError(f"Unsupported policy syntax: {type(node).__name__}")
        if isinstance(node, ast.Name | ast.arg | ast.FunctionDef | ast.Attribute):
            name = getattr(node, "id", getattr(node, "arg", getattr(node, "name", getattr(node, "attr", ""))))
            if name.startswith("_"):
                raise PolicyError("Private names and reflection are forbidden")
        if isinstance(node, ast.FunctionDef) and node.decorator_list:
            raise PolicyError("Decorators are forbidden")
        if isinstance(node, ast.arguments) and (node.vararg or node.kwarg or node.kwonlyargs):
            raise PolicyError("Variadic and keyword-only function parameters are unsupported")
        if isinstance(node, ast.comprehension) and node.is_async:
            raise PolicyError("Async comprehensions are forbidden")
    return tree


class ArtifactReader:
    """Provide the only artifact capabilities given to adapter programs.

    Instantiated once per pair side inside the worker subprocess. Generated
    adapter code reaches the filesystem only through the methods bound into
    ``capabilities()``; every read is confined to the owned side root.
    """

    def __init__(self, root: Path):
        """Bind the owned side root and start the observed-paths record.

        Called by _worker_pair for sides a and b, with a real copied directory.

        Args:
            root (Path): Absolute path of this side's artifact copy.

        Returns:
            None.
        """
        self.root = root.resolve()
        self.observed = set()

    def path(self, name: str) -> Path:
        """Confine an untrusted relative path to the owned root.

        Called by every read method with a path written by generated code.
        Rejects absolute paths, ``..`` traversal, and any component that is or
        passes through a symbolic link.

        Args:
            name (str): Relative artifact path, as written by the adapter.

        Returns:
            Path: Resolved file path inside the owned root.

        Raises:
            ValueError: For absolute, traversal, or symlink paths.
        """
        if not isinstance(name, str) or not name or Path(name).is_absolute():
            raise ValueError("Artifact reads require a relative path")
        if ".." in Path(name).parts:
            raise ValueError("Parent path traversal is forbidden")
        target = self.root
        for part in Path(name).parts:
            target /= part
            if target.is_symlink():
                raise ValueError("Artifact symlinks are forbidden")
        target.resolve().relative_to(self.root)
        return target

    def read_bytes(self, name: str) -> bytes:
        """Read one bounded artifact as raw bytes.

        Called by generated adapter code and by the other read methods. Refuses
        non-regular files and files over READ_BYTES, then re-checks after the
        read so a file that grew mid-read cannot leak unbounded bytes.

        Args:
            name (str): Relative artifact path.

        Returns:
            bytes: Full file content, at most READ_BYTES.

        Raises:
            ValueError: For non-regular, oversized, or growing artifacts.
        """
        target = self.path(name)
        if not target.is_file() or target.stat().st_size > READ_BYTES:
            raise ValueError("Artifact is not a regular file or exceeds read limit")
        data = target.read_bytes()
        if len(data) > READ_BYTES:
            raise ValueError("Artifact grew beyond the read limit")
        self.observed.add(Path(name).as_posix())
        return data

    def read_text(self, name: str) -> str:
        """Read one artifact as strict UTF-8 text.

        Called by generated adapter code for logs and reports. Decoding is
        strict, so binary content raises instead of producing mojibake values.

        Args:
            name (str): Relative artifact path.

        Returns:
            str: Decoded file content.

        Raises:
            UnicodeDecodeError: When the artifact is not valid UTF-8.
        """
        return self.read_bytes(name).decode("utf-8")

    def read_json(self, name: str):
        """Read and parse one JSON artifact.

        Called by generated adapter code for structured outputs such as
        metrics.json. Any JSON value is accepted, including lists and scalars.

        Args:
            name (str): Relative artifact path.

        Returns:
            The parsed JSON value (dict, list, str, number, bool, or None).
        """
        return json.loads(self.read_text(name))

    def read_csv(self, name: str, delimiter: str = ",") -> list[dict]:
        """Read and parse one CSV artifact into row dictionaries.

        Called by generated adapter code for tabular deliverables. The first
        row supplies the keys; all values are strings. A UTF-8 byte-order mark
        is dropped, so a header key is the real column name ("SMILES", not
        "\\ufeffSMILES") when adapters compare output names with input names.

        Args:
            name (str): Relative artifact path.
            delimiter (str): Field separator, a single character.

        Returns:
            list[dict]: One dictionary per data row, keys from the header.
        """
        text = self.read_bytes(name).decode("utf-8-sig")
        return list(csv.DictReader(io.StringIO(text), delimiter=delimiter))

    def exists(self, name: str) -> bool:
        """Check one path for existence and record the check as evidence.

        Called by generated adapter code to prove a required deliverable is
        present or absent. The check is recorded even when it returns False, so
        the verdict validator accepts it as grounded evidence.

        Args:
            name (str): Relative artifact path.

        Returns:
            bool: True when a regular file exists at the confined path.
        """
        result = self.path(name).is_file()
        self.observed.add(Path(name).as_posix())
        return result

    def list_files(self) -> list[str]:
        """List every regular artifact below the owned root.

        Called by generated adapter code to discover deliverables whose exact
        name the recon preview did not show. Symbolic links are never followed
        or listed.

        Args:
            None.

        Returns:
            list[str]: Sorted relative paths of all regular files.
        """
        paths = []
        for directory, directories, files in os.walk(self.root, followlinks=False):
            directories[:] = sorted(name for name in directories if not (Path(directory) / name).is_symlink())
            paths.extend((Path(directory) / name).relative_to(self.root).as_posix()
                         for name in files if not (Path(directory) / name).is_symlink())
        return sorted(paths)

    def capabilities(self) -> dict:
        """Bind the read methods exposed to generated code.

        Called by _worker_pair before running an adapter. Only these six
        functions enter the interpreter's name table; nothing else from this
        object is reachable.

        Args:
            None.

        Returns:
            dict: Method name to bound method, reads only.
        """
        return {name: getattr(self, name) for name in
                ("read_bytes", "read_text", "read_json", "read_csv", "exists", "list_files")}


class _Closure:
    """A generated function plus the scope it was defined in.

    Created by Interpreter.statement for every def statement; nested functions
    therefore keep access to the enclosing scope, like Python closures.
    """

    def __init__(self, node: ast.FunctionDef, scope):
        """Bind one def to its defining scope.

        Args:
            node (ast.FunctionDef): The def statement's AST node.
            scope (mapping): The scope in which the def appeared.

        Returns:
            None.
        """
        self.node = node
        self.scope = scope


class Interpreter:
    """Evaluate a small Python subset with no arbitrary host object access.

    Instantiated by _compare_once and the adapter step inside the worker for
    each untrusted source. Supports functions, assignments, if/elif/else, for,
    comprehensions, f-strings, and the allowlisted builtins, methods, and
    module calls; everything else raises PolicyError.
    """

    def __init__(self, source: str, reads: dict | None = None):
        """Screen the source and execute its top-level definitions.

        Called immediately after construction, on screened generated code.

        Args:
            source (str): Generated Python source.
            reads (dict | None): Named artifact-read functions for adapters.

        Returns:
            None.

        Raises:
            PolicyError: For syntax outside the allowed subset.
        """
        self.globals = {}
        self.reads = reads or {}
        self.block(screen_source(source).body, self.globals)

    def invoke(self, name: str, args: list, kwargs: dict | None = None):
        """Call one top-level function declared by the generated source.

        Called by _compare_once for ``compare`` and by the adapter step for
        ``normalize``. The function must exist at the top level; nested
        functions are reachable only from their enclosing scope.

        Args:
            name (str): Top-level function name.
            args (list): Positional argument values.
            kwargs (dict | None): Keyword argument values.

        Returns:
            The function's interpreted return value, JSON-compatible data.
        """
        closure = self.globals.get(name)
        if not isinstance(closure, _Closure):
            raise PolicyError(f"Missing function: {name}")
        return self.apply(closure, (args, kwargs or {}))

    def apply(self, closure: _Closure, call: tuple):
        """Run a function body in a scope that sees its defining scope.

        Called by invoke and by self.call for nested helper functions. Binds
        positional arguments, then defaults, then keyword arguments, mirroring
        Python's binding order.

        Args:
            closure (_Closure): Function node plus the scope it was defined in.
            call (tuple): ``(args list, kwargs dict)`` for this invocation.

        Returns:
            The interpreted return value, or None without an explicit return.
        """
        definition, (args, kwargs) = closure.node, call
        names = [parameter.arg for parameter in definition.args.posonlyargs + definition.args.args]
        if len(args) > len(names):
            raise PolicyError("Too many function arguments")
        own = dict(zip(names, args, strict=False))
        local = ChainMap(own, closure.scope)
        defaults = definition.args.defaults
        for key, value in zip(names[len(names) - len(defaults):], defaults, strict=True):
            own.setdefault(key, self.expression(value, closure.scope))
        for key, value in kwargs.items():
            if key not in names or key in names[:len(args)]:
                raise PolicyError("Unknown or duplicate keyword argument")
            own[key] = value
        if not set(names) <= set(own):
            raise PolicyError("Missing function arguments")
        try:
            self.block(definition.body, local)
        except _Return as result:
            return result.args[0]
        return None

    def block(self, statements: list, local: dict) -> None:
        """Execute a list of generated statements in order.

        Called by apply for function bodies and recursively by statement for
        if and loop bodies. Stops early only through the _Return, _Break, and
        _Continue control-flow exceptions.

        Args:
            statements (list[ast.stmt]): Statement nodes to execute.
            local (dict): Scope for name lookups and assignments.

        Returns:
            None.
        """
        for statement in statements:
            self.statement(statement, local)

    def bind(self, target: ast.AST, value, local: dict) -> None:
        """Assign one value to a generated assignment or loop target.

        Called by statement and loop. Supports names, tuple and list
        unpacking with a strict element count, and subscript stores.

        Args:
            target (ast.AST): Assignment target node.
            value: The evaluated right-hand side.
            local (dict): Scope to write into.

        Returns:
            None.

        Raises:
            PolicyError: For an unsupported target form.
        """
        if isinstance(target, ast.Name):
            local[target.id] = value
        elif isinstance(target, ast.Tuple | ast.List):
            for part, item in zip(target.elts, value, strict=True):
                self.bind(part, item, local)
        elif isinstance(target, ast.Subscript):
            self.expression(target.value, local)[self.expression(target.slice, local)] = value
        else:
            raise PolicyError("Unsupported assignment target")

    def statement(self, node: ast.AST, local: dict) -> None:
        """Execute one generated statement.

        Called by block for every node in a body. Function definitions become
        closures bound in the defining scope; break and continue unwind
        through the loop handler.

        Args:
            node (ast.AST): Statement node.
            local (dict): Scope for lookups, assignments, and def binding.

        Returns:
            None.

        Raises:
            PolicyError: For a statement kind outside the allowed subset.
        """
        if isinstance(node, ast.FunctionDef):
            local[node.name] = _Closure(node, local)
        elif isinstance(node, ast.Assign):
            value = self.expression(node.value, local)
            for target in node.targets:
                self.bind(target, value, local)
        elif isinstance(node, ast.AugAssign):
            self.bind(node.target, BINARY[type(node.op)](self.expression(node.target, local), self.expression(node.value, local)), local)
        elif isinstance(node, ast.Expr):
            self.expression(node.value, local)
        elif isinstance(node, ast.Return):
            raise _Return(self.expression(node.value, local) if node.value else None)
        elif isinstance(node, ast.If):
            self.block(node.body if self.expression(node.test, local) else node.orelse, local)
        elif isinstance(node, ast.For):
            self.loop(node, local)
        elif isinstance(node, ast.Raise):
            raise self.expression(node.exc, local)
        elif isinstance(node, ast.Break):
            raise _Break()
        elif isinstance(node, ast.Continue):
            raise _Continue()
        elif not isinstance(node, ast.Pass):
            raise PolicyError(f"Unsupported statement: {type(node).__name__}")

    def loop(self, node: ast.For, local: dict) -> None:
        """Execute a generated for loop over an evaluated iterable.

        Called by statement. A generated break unwinds as _Break and ends the
        loop; continue unwinds as _Continue and starts the next iteration.

        Args:
            node (ast.For): The for statement's AST node.
            local (dict): Scope for the iterable expression and the body.

        Returns:
            None.
        """
        for value in self.expression(node.iter, local):
            self.bind(node.target, value, local)
            try:
                self.block(node.body, local)
            except _Break:
                return
            except _Continue:
                continue
        self.block(node.orelse, local)

    def lookup(self, name: str, local: dict):
        """Resolve a name from locals, the defining scope, then value builtins.

        Called by expression for every generated name read. Mirrors Python's
        closure chain: the innermost scope wins, and only the harmless
        VALUE_NAMES builtins are reachable as plain values.

        Args:
            name (str): Identifier read by generated code.
            local (dict): Innermost scope (a ChainMap for function calls).

        Returns:
            The bound value.

        Raises:
            PolicyError: When no scope defines the name.
        """
        for scope in (local, self.globals, VALUE_NAMES):
            if name in scope:
                return scope[name]
        raise PolicyError(f"Undefined name: {name}")

    def expression(self, node: ast.AST, local: dict):
        """Evaluate one generated expression to a value.

        Called for every value position in the program: operands, call
        arguments, tests, subscripts, and f-string parts. Covers constants,
        names, collections, subscripts, slices, arithmetic, boolean and
        comparison chains, conditional expressions, calls, comprehensions,
        and f-strings.

        Args:
            node (ast.AST): Expression node.
            local (dict): Scope for name resolution.

        Returns:
            The evaluated value, JSON-compatible data.

        Raises:
            PolicyError: For an expression kind outside the allowed subset.
        """
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return self.lookup(node.id, local)
        if isinstance(node, ast.List | ast.Tuple | ast.Set):
            return {ast.List: list, ast.Tuple: tuple, ast.Set: set}[type(node)](self.expression(item, local) for item in node.elts)
        if isinstance(node, ast.Dict):
            return {self.expression(key, local): self.expression(value, local) for key, value in zip(node.keys, node.values, strict=True)}
        if isinstance(node, ast.Subscript):
            return self.expression(node.value, local)[self.expression(node.slice, local)]
        if isinstance(node, ast.Slice):
            return slice(*(self.expression(value, local) if value else None for value in (node.lower, node.upper, node.step)))
        if isinstance(node, ast.BinOp):
            return BINARY[type(node.op)](self.expression(node.left, local), self.expression(node.right, local))
        if isinstance(node, ast.UnaryOp):
            return {ast.Not: operator.not_, ast.USub: operator.neg, ast.UAdd: operator.pos}[type(node.op)](self.expression(node.operand, local))
        if isinstance(node, ast.IfExp):
            return self.expression(node.body if self.expression(node.test, local) else node.orelse, local)
        if isinstance(node, ast.Call):
            return self.call(node, local)
        if isinstance(node, ast.BoolOp | ast.Compare):
            return self.condition(node, local)
        if isinstance(node, ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp):
            return self.comprehension(node, local)
        if isinstance(node, ast.JoinedStr):
            return "".join(str(self.expression(part, local)) for part in node.values)
        if isinstance(node, ast.FormattedValue):
            return self.formatted(node, local)
        raise PolicyError(f"Unsupported expression: {type(node).__name__}")

    def condition(self, node: ast.AST, local: dict):
        """Evaluate a boolean or comparison chain with Python's short-circuit rules.

        Called by expression for BoolOp and Compare nodes. And and or return
        the deciding operand itself; comparison chains evaluate each pair in
        order and stop at the first false comparison.

        Args:
            node (ast.AST): BoolOp or Compare node.
            local (dict): Scope for operand evaluation.

        Returns:
            The deciding operand value for BoolOp, else a bool for Compare.
        """
        if isinstance(node, ast.BoolOp):
            for part in node.values:
                value = self.expression(part, local)
                if (isinstance(node.op, ast.And) and not value) or (isinstance(node.op, ast.Or) and value):
                    return value
            return value
        left = self.expression(node.left, local)
        for operation, part in zip(node.ops, node.comparators, strict=True):
            right = self.expression(part, local)
            if not COMPARE[type(operation)](left, right):
                return False
            left = right
        return True

    def call(self, node: ast.Call, local: dict):
        """Evaluate one call to a function, builtin, module, or method.

        Called by expression. Resolves plain names to generated closures, the
        adapter's read functions, or allowlisted builtins; attribute calls are
        restricted to the allowlisted json, math, and struct module functions
        and the METHODS table per value type. Everything else is forbidden.

        Args:
            node (ast.Call): Call node with evaluated-argument positions.
            local (dict): Scope for the callee name and arguments.

        Returns:
            The call's result value.

        Raises:
            PolicyError: For forbidden functions, methods, or call forms.
        """
        args = [self.expression(value, local) for value in node.args]
        kwargs = {item.arg: self.expression(item.value, local) for item in node.keywords}
        if None in kwargs:
            raise PolicyError("Expanded keyword arguments are unsupported")
        if isinstance(node.func, ast.Name):
            name = node.func.id
            target = local.get(name, self.globals.get(name))
            if isinstance(target, _Closure):
                return self.apply(target, (args, kwargs))
            function = self.reads.get(name) or BUILTINS.get(name)
            if function is None:
                raise PolicyError(f"Forbidden function: {name}")
            return function(*args, **kwargs)
        if isinstance(node.func, ast.Attribute):
            owner, name = node.func.value, node.func.attr
            if isinstance(owner, ast.Name) and owner.id in MODULE_CALLS:
                function = MODULE_CALLS[owner.id].get(name)
                if function is None:
                    raise PolicyError("Forbidden module operation")
                return function(*args, **kwargs)
            value = self.expression(owner, local)
            if name not in METHODS.get(type(value), set()):
                raise PolicyError(f"Forbidden method: {name}")
            return getattr(value, name)(*args, **kwargs)
        raise PolicyError("Indirect function calls are forbidden")

    def comprehension(self, node: ast.AST, local: dict):
        """Evaluate a list, set, dict comprehension, or generator expression.

        Called by expression. Iterates each generator in order, applies the
        if-filters, and collects the element expression over the surviving
        scopes, mirroring Python's comprehension nesting.

        Args:
            node (ast.AST): Comprehension or generator expression node.
            local (dict): Enclosing scope visible inside the comprehension.

        Returns:
            list, set, or dict: The collected result in the comprehension's type.
        """
        scopes = [dict(local)]
        for generator in node.generators:
            extended = []
            for scope in scopes:
                for value in self.expression(generator.iter, scope):
                    child = dict(scope)
                    self.bind(generator.target, value, child)
                    if all(self.expression(test, child) for test in generator.ifs):
                        extended.append(child)
            scopes = extended
        if isinstance(node, ast.DictComp):
            return {self.expression(node.key, scope): self.expression(node.value, scope) for scope in scopes}
        values = [self.expression(node.elt, scope) for scope in scopes]
        return set(values) if isinstance(node, ast.SetComp) else values

    def formatted(self, node: ast.FormattedValue, local: dict) -> str:
        """Render one f-string field exactly like Python.

        Called by expression for FormattedValue nodes inside JoinedStr
        f-strings. Applies the ``!s``/``!r``/``!a`` conversion, then the
        format specification, which may itself contain nested fields.

        Args:
            node (ast.FormattedValue): One parsed f-string field.
            local (dict): Scope for the value and format-spec expressions.

        Returns:
            str: The rendered field text.
        """
        value = CONVERSIONS[node.conversion](self.expression(node.value, local))
        spec = self.expression(node.format_spec, local) if node.format_spec else ""
        return format(value, spec)


def _validate_json(value) -> None:
    """Reject worker output that is not strict, finite JSON.

    Called on adapter results and verdicts inside the worker before any value
    is accepted. NaN and infinity would otherwise flow into the registry and
    break JSON round-trips.

    Args:
        value: The object about to be accepted as worker output.

    Returns:
        None.

    Raises:
        ValueError: When the value cannot be serialized without NaN or infinity.
    """
    json.dumps(value, allow_nan=False)


def _validate_normalized(value: dict, reader: ArtifactReader, criteria: list[dict]) -> None:
    """Require adapter output to cover the frozen criteria with real evidence.

    Called inside the worker right after an adapter runs. Every criterion must
    have a key (null allowed for unknown), every non-null value must be backed
    by evidence whose path the reader actually observed, and every evidence
    entry must name a known criterion with a value and a detail string.

    Args:
        value (dict): The adapter's returned ``{"values", "evidence"}`` object.
        reader (ArtifactReader): The side's reader, holding observed paths.
        criteria (list[dict]): The frozen criteria definitions.

    Returns:
        None.

    Raises:
        AdaptationError: For any shape, coverage, or grounding violation.
    """
    _validate_json(value)
    ids = {item["id"] for item in criteria}
    if not isinstance(value, dict) or not isinstance(value.get("values"), dict):
        raise AdaptationError("Adapter must return values and evidence")
    if set(value["values"]) != ids or not isinstance(value.get("evidence"), list):
        raise AdaptationError("Adapter keys must match the frozen criteria")
    supported = set()
    for evidence in value["evidence"]:
        if not isinstance(evidence, dict) or evidence.get("criterion") not in ids:
            raise AdaptationError("Adapter evidence needs a known criterion")
        if evidence.get("path") not in reader.observed:
            raise AdaptationError("Evidence path was not actually read or checked")
        if "value" not in evidence or not isinstance(evidence.get("detail"), str):
            raise AdaptationError("Evidence needs an observed value and detail")
        supported.add(evidence["criterion"])
    if any(item is not None and key not in supported for key, item in value["values"].items()):
        raise AdaptationError("Non-null criterion has no artifact evidence")
    if all(item is None for item in value["values"].values()):
        raise AdaptationError("Adapter measured nothing for an executed generation: "
                              "the artifact schema needs a new adapter, not a null verdict")
    _check_directional_numbers(value["values"], criteria)


def _check_directional_numbers(values: dict, criteria: list[dict]) -> None:
    """Require a number or null for every "higher" or "lower" criterion.

    Called by ``_validate_normalized``. A boolean value can never differ
    usefully: in the drugex replay (2026-10-18) "visualization_quality"
    (higher) was True for every run, so every pair tied.

    Args:
        values (dict): Adapter values by criterion id.
        criteria (list[dict]): The frozen criteria definitions.

    Returns:
        None.

    Raises:
        AdaptationError: When a directional value is not a number.
    """
    directions = {criterion["id"]: criterion.get("direction") for criterion in criteria}
    for key, item in values.items():
        if directions.get(key) in {"higher", "lower"} and item is not None and (isinstance(item, bool) or not isinstance(item, int | float)):
            raise AdaptationError(f"Criterion '{key}' is {directions[key]}-is-better: its value must be a number "
                                  f"or null, not {type(item).__name__} {item!r}")


def _validate_verdict(verdict: dict, normalized: list[dict]) -> None:
    """Validate a comparator verdict against the public pair contract.

    Called on every comparator run inside the worker, including both orders
    and self-tie smoke checks. A decisive winner needs a positive stage and at
    least one evidence entry, and every evidence entry must be copied
    unchanged from the adapter output of the side it claims.

    Args:
        verdict (dict): The comparator's ``{"winner", "stage", ...}`` object.
        normalized (list[dict]): Adapter outputs for sides a and b, in order.

    Returns:
        None.

    Raises:
        PolicyError: For any contract violation.
    """
    _validate_json(verdict)
    if not isinstance(verdict, dict) or verdict.get("winner") not in {"a", "b", "tie"}:
        raise PolicyError("Verdict winner must be a, b or tie")
    if type(verdict.get("stage")) is not int or verdict["stage"] < 0:
        raise PolicyError("Verdict stage must be a nonnegative integer")
    if any(not isinstance(verdict.get(key), str) or not verdict[key].strip() for key in ("feedback_a", "feedback_b")):
        raise PolicyError("Both sides need feedback strings")
    evidence = verdict.get("evidence")
    if not isinstance(evidence, list):
        raise PolicyError("Verdict evidence must be a list")
    if verdict["winner"] != "tie" and (not evidence or verdict["stage"] < 1):
        raise PolicyError("A decisive verdict requires evidence and a positive stage")
    for entry in evidence:
        if not isinstance(entry, dict) or entry.get("side") not in {"a", "b"}:
            raise PolicyError("Verdict evidence needs a valid side")
        source = normalized[0 if entry["side"] == "a" else 1]["evidence"]
        if {key: value for key, value in entry.items() if key != "side"} not in source:
            raise PolicyError("Verdict evidence was not produced by the adapter")


def _compare_once(source: str, pair: list[dict]) -> dict:
    """Run the comparator once on deep copies and validate its verdict.

    Called by _worker_pair for the forward order, the reversed order, the two
    self-tie checks, and the determinism re-run. Deep copies guarantee the
    comparator cannot mutate the normalized values it is judging.

    Args:
        source (str): The frozen comparison policy source.
        pair (list[dict]): Adapter outputs for sides a and b, in order.

    Returns:
        dict: A verdict that satisfies _validate_verdict.
    """
    untouched = copy.deepcopy(pair)
    verdict = Interpreter(source).invoke("compare", copy.deepcopy(pair))
    _validate_verdict(verdict, untouched)
    return verdict


def _check_symmetry(forward: dict, backward: dict) -> None:
    """Reject a comparator whose judgment depends on argument order.

    Called once per real pair inside the worker. The reversed-order run must
    flip the winner, keep the stage, swap the two feedback strings exactly,
    and mirror the evidence entries with swapped side labels.

    Args:
        forward (dict): Verdict from compare(a, b).
        backward (dict): Verdict from compare(b, a).

    Returns:
        None.

    Raises:
        PolicyError: When any mirrored field disagrees.
    """
    flip = {"a": "b", "b": "a", "tie": "tie"}
    if forward["winner"] != flip[backward["winner"]] or forward["stage"] != backward["stage"]:
        raise PolicyError("Comparator violates winner/stage symmetry")
    if forward["feedback_a"] != backward["feedback_b"] or forward["feedback_b"] != backward["feedback_a"]:
        raise PolicyError("Comparator violates feedback symmetry")
    mirrored = [{**entry, "side": flip[entry["side"]]} for entry in backward["evidence"]]
    if sorted(forward["evidence"], key=_canonical) != sorted(mirrored, key=_canonical):
        raise PolicyError("Comparator violates evidence symmetry")


def _canonical(entry: dict) -> str:
    """Give one evidence entry a stable sort key for symmetric comparison.

    Called by _check_symmetry, twice per pair, on evidence lists whose
    ordering the comparator is free to choose.

    Args:
        entry (dict): One evidence entry with a side label.

    Returns:
        str: Sorted-key JSON text usable as a sort key.
    """
    return json.dumps(entry, sort_keys=True)


def _worker_pair(payload: dict) -> dict:
    """Normalize both real sides, then judge the pair under all guard checks.

    Called by _worker_main, the worker subprocess entry point, on the JSON
    payload from run_pair. Runs each side's adapter against its real files,
    compares in both argument orders, verifies both self-ties, re-runs the
    forward order to prove determinism, and rejects ties that ignore a known
    difference.

    Args:
        payload (dict): ``{"root", "core_source", "criteria", "adapters"}``
            with root holding a/ and b/ artifact copies.

    Returns:
        dict: The forward verdict with channel "code" and the two sides'
        normalized values under ``normalized_a`` and ``normalized_b``.
    """
    normalized = []
    for side in ("a", "b"):
        reader = ArtifactReader(Path(payload["root"]) / side)
        try:
            value = Interpreter(payload["adapters"][side], reader.capabilities()).invoke("normalize", [])
            _validate_normalized(value, reader, payload["criteria"])
        except Exception as error:
            raise AdaptationError(f"Adapter {side}: {type(error).__name__}: {error}") from error
        normalized.append(value)
    source = payload["core_source"]
    forward = _compare_once(source, normalized)
    backward = _compare_once(source, list(reversed(normalized)))
    _check_symmetry(forward, backward)
    for value in normalized:
        if _compare_once(source, [value, value])["winner"] != "tie":
            raise PolicyError("Comparator failed its real-artifact self-tie check")
    if _compare_once(source, normalized) != forward:
        raise PolicyError("Comparator is not deterministic")
    _check_decides_differences(forward, normalized, payload["criteria"])
    _check_null_never_decides(forward, normalized, payload["criteria"])
    return {**forward, "channel": "code", "normalized_a": normalized[0], "normalized_b": normalized[1]}


def _check_decides_differences(verdict: dict, normalized: list[dict], criteria: list[dict]) -> None:
    """Reject a tie that ignores a known difference on a directional criterion.

    Called once per real pair inside the worker. It catches comparators that
    read the wrong keys, as one live HP-folding run did when it tied -9
    against -8. A failed "satisfy" gate on either side may legitimately void
    measurements, so the check only applies when no gate failed.

    Args:
        verdict (dict): Forward comparator verdict.
        normalized (list[dict]): Adapter outputs for sides a and b, in order.
        criteria (list[dict]): The frozen criteria definitions.

    Returns:
        None.

    Raises:
        PolicyError: When a higher/lower criterion has two known, different
            values but the verdict is a tie.
    """
    left, right = (side["values"] for side in normalized)
    gates_failed = any(criterion["direction"] == "satisfy"
                       and (left[criterion["id"]] is False or right[criterion["id"]] is False)
                       for criterion in criteria)
    if verdict["winner"] != "tie" or gates_failed:
        return
    for criterion in criteria:
        key = criterion["id"]
        known = left[key] is not None and right[key] is not None
        if known and left[key] != right[key] and criterion["direction"] in {"higher", "lower"}:
            raise PolicyError(f"Comparator tied although '{key}' differs ({left[key]!r} vs {right[key]!r}). "
                              "Read values as a['values'][criterion_id] and let a known difference on a "
                              "higher/lower criterion decide that criterion.")


def _check_null_never_decides(verdict: dict, normalized: list[dict], criteria: list[dict]) -> None:
    """Reject a win decided on a stage where a value is unknown.

    Null means unknown, not a measured failure. Called once per real pair
    inside the worker. A live Ames replay (2026-10-18) let the one run whose
    printed validation AUC the adapter parsed beat six runs whose logs it
    failed to parse, and told them they "provided no" metric. A win is valid
    only when at least one criterion at the deciding stage is known on both
    sides and differs; a known sibling at the same stage does not license a
    decision on a null one.

    Args:
        verdict (dict): Forward comparator verdict.
        normalized (list[dict]): Adapter outputs for sides a and b, in order.
        criteria (list[dict]): The frozen criteria definitions.

    Returns:
        None.

    Raises:
        PolicyError: When no criterion at the deciding stage has two known,
        different values.
    """
    if verdict["winner"] == "tie":
        return
    left, right = (side["values"] for side in normalized)
    staged = [c["id"] for c in criteria if c.get("stage") == verdict["stage"]]
    decidable = [key for key in staged if None not in (left.get(key), right.get(key)) and left[key] != right[key]]
    if staged and not decidable:
        raise PolicyError(f"Comparator decided stage {verdict['stage']} although no criterion there "
                          f"({', '.join(staged)}) has two known, different values. Null is not a loss: "
                          "if either value is None, tie that criterion and continue to the next stage.")


def _worker_main() -> None:
    """Serve one comparison request in the isolated worker subprocess.

    Called as ``policy.py --worker`` by run_pair, with the request JSON on
    stdin and resource limits already applied (file size, open files, and
    address space outside macOS). Prints one JSON envelope: ``{"ok": True,
    "result": ...}`` on success, ``{"ok": False, "error", "kind"}`` with kind
    "adapter" or "policy" on failure, so the parent can route repairs.

    Args:
        None. The request arrives on stdin.

    Returns:
        None. The response envelope goes to stdout.
    """
    try:
        resource.setrlimit(resource.RLIMIT_FSIZE, (WORKER_BYTES, WORKER_BYTES))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        if sys.platform != "darwin":
            resource.setrlimit(resource.RLIMIT_AS, (MEMORY_BYTES, MEMORY_BYTES))
        payload = json.loads(sys.stdin.buffer.read(WORKER_BYTES))
        result = {"ok": True, "result": _worker_pair(payload)}
    except Exception as error:
        result = {"ok": False, "error": f"{type(error).__name__}: {error}",
                  "kind": "adapter" if isinstance(error, AdaptationError) else "policy"}
    print(json.dumps(result, allow_nan=False))


def run_pair(payload: dict, timeout: float) -> dict:
    """Run one comparison in a fresh worker subprocess with a hard deadline.

    Called by PolicyEngine._compare_repair for every attempt, real pair or
    smoke pair. The worker runs with ``-I -S``, a fixed hash seed, an empty
    environment, and its own process group, so the comparison cannot see host
    state or outlive its timeout.

    Args:
        payload (dict): ``{"root", "core_source", "criteria", "adapters"}``;
            must stay under WORKER_BYTES when serialized.
        timeout (float): Seconds before the worker's process group is killed.

    Returns:
        dict: The validated verdict from _worker_pair.

    Raises:
        PolicyError: For an oversized request, timeout, worker crash,
            unparseable response, or a comparator fault.
        AdaptationError: When the worker reports an adapter fault.
    """
    encoded = json.dumps(payload, allow_nan=False).encode()
    if len(encoded) > WORKER_BYTES:
        raise PolicyError("Worker request exceeds byte limit")
    with tempfile.TemporaryFile() as output:
        process = subprocess.Popen([sys.executable, "-I", "-S", str(Path(__file__).resolve()), "--worker"],
                                   stdin=subprocess.PIPE, stdout=output, stderr=output,
                                   cwd=payload["root"], env={"PYTHONHASHSEED": "0"}, start_new_session=True)
        try:
            process.communicate(encoded, timeout=timeout)
        except subprocess.TimeoutExpired as error:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            raise PolicyError(f"Policy worker timed out after {timeout} seconds") from error
        output.seek(0)
        raw = output.read(WORKER_BYTES + 1)
    if process.returncode or len(raw) > WORKER_BYTES:
        raise PolicyError(f"Policy worker failed (exit {process.returncode}); {raw[:2000]!r}")
    try:
        response = json.loads(raw)
    except (ValueError, UnicodeError) as error:
        raise PolicyError(f"Invalid worker response: {raw[:2000]!r}") from error
    if not response.get("ok"):
        error_type = AdaptationError if response.get("kind") == "adapter" else PolicyError
        raise error_type(response.get("error", "Unspecified worker error"))
    return response["result"]


def bundle_version(bundle: dict) -> str:
    """Hash every input that can change a comparison verdict.

    Called after prepare and after every compare so the evaluator can detect
    that adapters or the comparison policy changed mid-tournament. Deliberately
    excludes runtime-only fields such as ``_pair_uuids`` and per-pair
    normalized values, which cannot change other pairs' verdicts.

    Args:
        bundle (dict): The prepared task bundle.

    Returns:
        str: SHA-256 hex digest over the sorted JSON of the semantic keys.
    """
    keys = ("prompt_version", "goal", "task_representation", "criteria", "representation",
            "core_source", "adapters", "assignments", "visuals")
    data = {key: bundle[key] for key in keys}
    return hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()


class PolicyEngine:
    """Keep goal criteria fixed while artifact-schema adapters expand.

    Created once per evaluator. The first eligible generation freezes the task
    representation, criteria, and comparison policy; every later generation
    only adds or repairs schema adapters, so the definition of success can
    never drift with observed performance.
    """

    def __init__(self, config, ask):
        """Bind the verifier settings and the text-model callback.

        Called by PolicyBTVerifierEvaluator at construction.

        Args:
            config (object | dict): Mimosa config or a plain dict with the
                ``policy_bt_*`` settings.
            ask (Callable[[str], str]): Callback sending one prompt to the
                configured text model and returning its raw response.

        Returns:
            None.
        """
        self.config = config
        self.ask = ask

    def option(self, name: str, default):
        """Read one ``policy_bt_*`` setting from either config form.

        Called throughout the engine for timeouts and repair-round counts, so
        both the real Config object and the tests' plain dicts work.

        Args:
            name (str): Setting name, for example ``policy_bt_scorer_timeout_s``.
            default: Value used when the setting is absent.

        Returns:
            The configured value, or ``default``.
        """
        return self.config.get(name, default) if isinstance(self.config, dict) else getattr(self.config, name, default)

    def source(self, instructions: str, evidence: dict) -> str:
        """Ask the model for generated source and screen it, with repair rounds.

        Called by prepare for the core policy and each new adapter, and by
        _compare_repair for repairs. A response that is not valid JSON, or
        whose source fails the language screen, is fed back to the model with
        the error until ``policy_bt_repair_rounds`` is exhausted.

        Args:
            instructions (str): The contract prompt (ADAPTER or CORE, plus
                REPAIR for repairs).
            evidence (dict): JSON-compatible request data embedded in the
                prompt after the instructions.

        Returns:
            str: Generated Python source that passes screen_source.

        Raises:
            PolicyError: When no round produces screenable source.
        """
        from .artifacts import decode_object
        from .prompts import REPAIR, evidence_prompt

        request = evidence
        for attempt in range(int(self.option("policy_bt_repair_rounds", 3)) + 1):
            response = self.ask(evidence_prompt(instructions, request))
            try:
                source = decode_object(response)["source"]
                screen_source(source)
                return source
            except (ValueError, KeyError, TypeError, SyntaxError) as error:
                if attempt == int(self.option("policy_bt_repair_rounds", 3)):
                    raise PolicyError(f"Source generation failed: {error}") from error
                request = {"original_request": evidence, "failed_response": response, "error": str(error)}
                instructions = REPAIR
        raise PolicyError("No source generation attempts configured")

    def prepare(self, goal: str, generations: list[dict], cached: dict | None = None) -> dict:
        """Freeze the task once, then extend adapters for each new schema.

        Called by the evaluator before every tournament refresh. With no cache
        it runs reconnaissance, freezes the task fields, and generates the
        comparison policy. With a cache it restores the frozen fields verbatim
        (a stale cache is discarded and re-frozen) and only plans adapters for
        generations that lack an assignment. Every generation is then smoked
        against itself before the bundle is hashed.

        Args:
            goal (str): The exact user task text.
            generations (list[dict]): Eligible generation records, each with
                ``uuid`` and ``snapshot`` keys.
            cached (dict | None): The previous persistable bundle; never
                modified.

        Returns:
            dict: The updated bundle with ``version``, ``core_source``,
            ``adapters``, ``assignments``, ``visuals``, and ``inspection``.

        Raises:
            PolicyError: When there is no eligible generation or any
                generation round fails after all repairs.
        """
        from .prompts import ADAPTER, CORE, PROMPT_VERSION
        from .recon import code_task, reconnaissance

        if not generations:
            raise PolicyError("Policy preparation requires an eligible generation")
        if cached and (cached.get("goal") != goal or cached.get("prompt_version") != PROMPT_VERSION):
            cached = None
        bundle = copy.deepcopy(cached) if cached else {"goal": goal, "prompt_version": PROMPT_VERSION,
                  "adapters": {}, "assignments": {}, "visuals": {}}
        frozen_keys = ("task_representation", "criteria", "representation")
        frozen = {key: bundle[key] for key in frozen_keys} if cached else None
        plan = reconnaissance(goal, generations, {"ask": self.ask, "frozen_task": frozen,
                                                 "adapters": bundle["adapters"]})
        if not cached:
            bundle.update({key: plan[key] for key in frozen_keys})
            # Freeze-time guard record: verdicts, revision requests and drops.
            bundle["criteria_guard"] = {key: plan["inspection"].get(key, []) for key in GUARD_RECORD_KEYS}
            bundle["core_source"] = self.source(CORE, {"goal": goal, **code_task({key: bundle[key] for key in frozen_keys})})
        for generation in generations:
            uuid = generation["uuid"]
            adapter_plan = plan["plans"][uuid]
            adapter_id = adapter_plan.get("adapter")
            if adapter_id not in bundle["adapters"]:
                source = self.source(ADAPTER, {"goal": goal, "frozen_task": code_task({key: bundle[key] for key in frozen_keys}),
                                     "plan": adapter_plan, "generation_uuid": uuid, "inspection": plan["inspection"]})
                adapter_id = hashlib.sha256(source.encode()).hexdigest()
                bundle["adapters"][adapter_id] = source
            bundle["assignments"][uuid] = adapter_id
            bundle["visuals"][uuid] = plan.get("visuals", {}).get(uuid, [])
        bundle["inspection"] = plan["inspection"]
        self._prepare_smoke(bundle, generations)
        bundle["version"] = bundle_version(bundle)
        return bundle

    def _prepare_smoke(self, bundle: dict, generations: list[dict]) -> None:
        """Exercise every new schema against itself before the tournament runs.

        Called by prepare for each generation being planned. Comparing a
        snapshot with itself must tie; running it here means an adapter or
        policy defect is found and repaired before any real predecessor
        comparison can produce a verdict.

        Args:
            bundle (dict): Prepared task, mutated by successful repairs.
            generations (list[dict]): Generation records with ``uuid`` and
                ``snapshot`` keys.

        Returns:
            None.

        Raises:
            PolicyError: When a smoke comparison fails after all repairs.
        """
        from .artifacts import pair_workspace

        for generation in generations:
            uuid = generation["uuid"]
            root = Path(generation["snapshot"])
            with pair_workspace(root, root) as pair_root:
                self._compare_repair(bundle, Path(pair_root), [uuid, uuid])

    def compare(self, bundle: dict, pair_root: Path) -> dict:
        """Compare one real a/b pair under the frozen task policy.

        Called by the evaluator for every pair that is not already cached, with
        ``bundle["_pair_uuids"]`` naming the generations whose copies sit at
        ``pair_root/a`` and ``pair_root/b``. A successful adapter repair inside
        this call changes the bundle, so the caller refreshes its version.

        Args:
            bundle (dict): Prepared task; repaired adapters are written back
                into it.
            pair_root (Path): Directory holding the two artifact copies.

        Returns:
            dict: The validated verdict, including ``channel`` and the two
            sides' normalized values.

        Raises:
            PolicyError: When no repair round produces a valid comparison.
        """
        uuids = bundle.get("_pair_uuids")
        if not isinstance(uuids, list) or len(uuids) != 2:
            raise PolicyError("compare requires bundle['_pair_uuids']=[a_uuid,b_uuid]")
        result = self._compare_repair(bundle, pair_root, uuids)
        bundle["version"] = bundle_version(bundle)
        return result

    def _compare_repair(self, bundle: dict, root: Path, uuids: list[str]) -> dict:
        """Repair adapter or comparison code while the criteria stay frozen.

        Called by compare and _prepare_smoke. Adapter faults trigger adapter
        repair for both sides; any other comparator fault triggers comparison
        repair. Each repair is itself generated under repair rounds, and the
        loop stops after ``policy_bt_repair_rounds`` failed attempts.

        Args:
            bundle (dict): Prepared task, mutated by successful repairs.
            root (Path): Directory holding the a/ and b/ artifact copies.
            uuids (list[str]): Generation identifiers at sides a and b.

        Returns:
            dict: A validated verdict from run_pair.

        Raises:
            PolicyError: When the last attempt still fails.
        """
        from .prompts import ADAPTER_CONTRACT, CORE_CONTRACT, REPAIR
        from .recon import code_task

        task = code_task(bundle)

        rounds = int(self.option("policy_bt_repair_rounds", 3))
        for attempt in range(rounds + 1):
            payload = {"root": str(root.resolve()), "core_source": bundle["core_source"], "criteria": task["criteria"],
                       "adapters": {side: bundle["adapters"][bundle["assignments"][uuid]]
                                    for side, uuid in zip(("a", "b"), uuids, strict=True)}}
            try:
                return run_pair(payload, float(self.option("policy_bt_scorer_timeout_s", 60)))
            except PolicyError as error:
                if attempt == rounds:
                    raise
                context = {"goal": bundle["goal"], "criteria": task["criteria"],
                           "representation": task["representation"], "error": str(error),
                           "inspection": bundle.get("inspection", {})}
                if not isinstance(error, AdaptationError):
                    bundle["core_source"] = self.source(REPAIR + CORE_CONTRACT, {**context, "kind": "comparison_policy", "source": bundle["core_source"]})
                    continue
                for uuid in dict.fromkeys(uuids):
                    old_id = bundle["assignments"][uuid]
                    source = self.source(REPAIR + ADAPTER_CONTRACT, {**context, "kind": "adapter", "uuid": uuid, "source": bundle["adapters"][old_id]})
                    new_id = hashlib.sha256(source.encode()).hexdigest()
                    bundle["adapters"][new_id] = source
                    bundle["assignments"][uuid] = new_id
        raise PolicyError("No comparison attempts configured")


if __name__ == "__main__":
    if sys.argv[1:] == ["--worker"]:
        _worker_main()
    else:
        print("PolicyEngine(config, ask).prepare(goal, generations, cached); compare(bundle, pair_root)")

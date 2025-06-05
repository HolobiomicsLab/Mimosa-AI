import os
import sys
import signal
import time
import contextlib
import traceback
import threading
from io import StringIO
from typing import Dict, Any, Optional, List
import logging


class ExecutionTimeout(Exception):
    """Raised when code execution exceeds timeout."""
    pass

class PySandbox:
    """
    A sandboxed Python code interpreter with execution control and safety features.
    Inspired by E2B but running locally with better control over the environment.
    """
    
    def __init__(self, 
                 timeout: int = 30, 
                 max_output_size: int = 1024 * 1024,  # 1MB
                 logger: Optional[logging.Logger] = None):
        self.timeout = timeout
        self.max_output_size = max_output_size
        self.logger = logger or self._setup_logger()
        self.global_vars = self._create_safe_globals()
        self.execution_count = 0
        
    def _setup_logger(self) -> logging.Logger:
        """Setup default logger if none provided."""
        logger = logging.getLogger(self.__class__.__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _create_safe_globals(self) -> Dict[str, Any]:
        """Create a controlled global environment."""
        # Start with minimal builtins
        safe_builtins = {
            # Safe built-ins
            'abs', 'all', 'any', 'bin', 'bool', 'bytearray', 'bytes',
            'chr', 'dict', 'dir', 'divmod', 'enumerate', 'filter',
            'float', 'format', 'frozenset', 'getattr', 'hasattr',
            'hash', 'hex', 'id', 'int', 'isinstance', 'issubclass',
            'iter', 'len', 'list', 'map', 'max', 'min', 'next',
            'oct', 'ord', 'pow', 'print', 'range', 'repr', 'reversed',
            'round', 'set', 'setattr', 'slice', 'sorted', 'str',
            'sum', 'tuple', 'type', 'vars', 'zip',
            # Exception types
            'Exception', 'ValueError', 'TypeError', 'KeyError',
            'IndexError', 'AttributeError', 'RuntimeError',
        }
        
        restricted_builtins = {name: getattr(__builtins__, name) 
                             for name in safe_builtins 
                             if hasattr(__builtins__, name)}
        
        return {
            '__builtins__': restricted_builtins,
            '__name__': '__main__',
            'math': __import__('math'),
            'json': __import__('json'),
            'datetime': __import__('datetime'),
            're': __import__('re'),
        }
    
    def add_tool(self, name: str, func: callable):
        """Add a tool function to the interpreter's global scope."""
        self.global_vars[name] = func
        self.logger.info(f"Added tool: {name}")
    
    def add_module(self, name: str, module):
        """Add a module to the interpreter's global scope."""
        self.global_vars[name] = module
        self.logger.info(f"Added module: {name}")
    
    def _timeout_handler(self, signum, frame):
        """Handle execution timeout."""
        raise ExecutionTimeout(f"Code execution exceeded {self.timeout} seconds")
    
    @contextlib.contextmanager
    def _capture_output(self):
        """Context manager to capture stdout and stderr."""
        old_stdout, old_stderr = sys.stdout, sys.stderr
        stdout_buffer = StringIO()
        stderr_buffer = StringIO()
        
        try:
            sys.stdout = stdout_buffer
            sys.stderr = stderr_buffer
            yield stdout_buffer, stderr_buffer
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
    
    def _run_code_with_timeout(self, code: str, globals_dict: Dict[str, Any]) -> Any:
        """run_code code with timeout protection."""
        result = None
        exception = None
        
        def target():
            nonlocal result, exception
            try:
                try:
                    result = eval(code, globals_dict)
                except SyntaxError:
                    exec(code, globals_dict)
                    result = None
            except Exception as e:
                exception = e
        
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout=self.timeout)
        
        if thread.is_alive():
            raise ExecutionTimeout(f"Code execution exceeded {self.timeout} seconds")
        if exception:
            raise exception
        return result
    
    def run_code(self, 
                code: str, 
                safety: bool = False, 
                verbose: bool = False,
                reset_globals: bool = False) -> Dict[str, Any]:
        """
        run_code Python code and return comprehensive results.
        
        Args:
            code: Python code to run_code
            safety: If True, prompt user for confirmation
            reset_globals: If True, reset global variables before execution
            
        Returns:
            Dict containing execution results, output, errors, etc.
        """
        if safety and input(f"run_code code?\n{code}\n(y/n): ").lower() != 'y':
            return {
                'success': False,
                'output': '',
                'error': 'Code execution rejected by user',
                'execution_count': self.execution_count
            }
        if reset_globals:
            self.global_vars = self._create_safe_globals()
        self.execution_count += 1
        code = code.strip()
        if not code:
            return {
                'success': False,
                'output': '',
                'error': 'No code provided',
                'execution_count': self.execution_count
            }
        
        self.logger.info(f"Executing code (#{self.execution_count}):\n{code}")
        with self._capture_output() as (stdout_buffer, stderr_buffer):
            try:
                result = self._run_code_with_timeout(code, self.global_vars)
                stdout_content = stdout_buffer.getvalue()
                stderr_content = stderr_buffer.getvalue()
                if len(stdout_content) > self.max_output_size:
                    stdout_content = stdout_content[:self.max_output_size] + "\n... (output truncated)"
                output_parts = []
                if stdout_content:
                    output_parts.append(stdout_content)
                if result is not None:
                    output_parts.append(repr(result))
                
                output = '\n'.join(output_parts)
                
                return {
                    'success': True,
                    'output': output,
                    'error': stderr_content if stderr_content else None,
                    'result': result,
                    'execution_count': self.execution_count
                }
            except ExecutionTimeout as e:
                error_msg = str(e)
                self.logger.warning(f"Code execution timed out: {error_msg}")
                return {
                    'success': False,
                    'output': stdout_buffer.getvalue(),
                    'error': error_msg,
                    'execution_count': self.execution_count
                }
            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                traceback_str = traceback.format_exc()
                print(f"Code execution failed:\n{error_msg}")
                print(f"Full traceback:\n{traceback_str}")
                return {
                    'success': False,
                    'output': stdout_buffer.getvalue(),
                    'error': error_msg,
                    'traceback': traceback_str,
                    'execution_count': self.execution_count
                }
    
    def reset(self):
        """Reset the interpreter state."""
        self.global_vars = self._create_safe_globals()
        self.execution_count = 0
    
    def get_variables(self) -> Dict[str, str]:
        """Get current variables in the global scope (for debugging)."""
        user_vars = {k: repr(v) for k, v in self.global_vars.items() 
                    if not k.startswith('__') and k not in ['math', 'json', 'datetime', 're']}
        return user_vars
    
    def install_package(self, package: str) -> Dict[str, Any]:
        """
        Install a package using pip (use cautiously).
        This breaks the sandbox but might be needed for some tools.
        """
        try:
            import subprocess
            result = subprocess.run([sys.executable, '-m', 'pip', 'install', package], 
                                  capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                return {'success': True, 'output': result.stdout}
            else:
                return {'success': False, 'error': result.stderr}
                
        except Exception as e:
            error_msg = f"Package installation failed: {str(e)}"
            return {'success': False, 'error': error_msg}
    
    def close(self):
        pass

if __name__ == "__main__":
    sandbox = PySandbox(timeout=10)
    test_code = """
x = 5 + 3
print(f"Math: {x}")
"""
    result = sandbox.run_code(test_code)
    print("Execution result:", result)
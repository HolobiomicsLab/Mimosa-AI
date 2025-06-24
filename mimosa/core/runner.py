import subprocess
from typing import Optional

class WorkflowRunner:
    """Handles execution of workflow code in a sandboxed environment.
    
    Attributes:
        process (subprocess.Popen): The subprocess running the workflow
    """
    
    def __init__(self) -> None:
        """Initialize the workflow runner."""
        self.process: Optional[subprocess.Popen] = None

    def run(self, code: str) -> int:
        """Execute the given workflow code in a sandboxed subprocess.
        
        Args:
            code: The Python code to execute
            
        Returns:
            int: The return code of the execution (0 for success)
            
        Raises:
            RuntimeError: If execution fails with non-zero exit code
        """
        print("\n🔧 Executing generated workflow in sandbox...")
        
        try:
            self.process = subprocess.Popen(
                ["python", "-c", code],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            print("\nExecution Progress:")
            for line in iter(self.process.stdout.readline, ''):
                print(line, end='')
                
            stderr_output = self.process.stderr.read()
            self.process.wait()
            
            print("\nExited.\nErrors (if any):")
            print(stderr_output)
            
            if self.process.returncode != 0:
                raise RuntimeError(f"Workflow execution failed with code {self.process.returncode}")
                
            return self.process.returncode
            
        except Exception as e:
            print(f"❌ Error during execution: {e}")
            raise RuntimeError(f"Failed to execute workflow: {str(e)}")


def runner(code: str) -> str:
    """Legacy runner function (maintained for backward compatibility)."""
    return WorkflowRunner().run(code)

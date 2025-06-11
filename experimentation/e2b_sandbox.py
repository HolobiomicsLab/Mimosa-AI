import os
from e2b_code_interpreter import Sandbox

class E2BSandbox:
    """A class to manage a sandbox environment for code execution."""
    
    def __init__(self, requirements_file: str = "./sandbox_requirements.txt", tools_folder: str = "./tools_client"):
        """Initialize the sandbox with paths to requirements and tools."""
        self.requirements_file = requirements_file
        self.tools_folder = tools_folder
        self.sandbox = self._create_sandbox()
    
    def _create_sandbox(self) -> Sandbox:
        """Create and configure the sandbox with required dependencies."""
        print("📦 Setting up sandbox environment...")
        sandbox = Sandbox()
        
        if not os.path.exists(self.requirements_file):
            raise FileNotFoundError(f"create_sandbox: Requirements file '{self.requirements_file}' not found.")
        
        with open(self.requirements_file, 'r') as f:
            requirements = f.read().strip().split('\n')
        
        for requirement in requirements:
            requirement = requirement.strip()
            if requirement and not requirement.startswith('#'):
                print(f"Installing {requirement}...")
                sandbox.commands.run(f"pip install {requirement}")
        
        return sandbox
    
    def run_code(self, code: str, verbose: bool = False) -> str:
        """Execute code in the sandbox and return output or raise errors."""
        if not os.path.exists(self.tools_folder):
            raise FileNotFoundError(f"run_code_sandbox: Tools folder '{self.tools_folder}' not found.")
        
        for root, dirs, files in os.walk(self.tools_folder):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r') as f:
                        file_content = f.read()
                    relative_path = os.path.relpath(file_path, '.')
                    print(f"Importing {relative_path} to sandbox...")
                    self.sandbox.files.write(relative_path, file_content)
        
        execution = self.sandbox.run_code(
            code,
            envs={'HF_TOKEN': os.getenv('HF_TOKEN')}
        )
        
        if execution.error:
            execution_logs = "\n".join([str(log) for log in execution.logs.stdout])
            logs = execution_logs + execution.error.traceback
            raise ValueError(logs)
        
        return "\n".join([str(log) for log in execution.logs.stdout])
    
    def close(self):
        """Close the sandbox to free resources."""
        if self.sandbox:
            self.sandbox.close()
            print("🛑 Sandbox closed.")

import shutil
from pathlib import Path
from typing import Union
from pathlib import Path
from sources.core.llm_provider import LLMProvider, LLMConfig
import re
import os

class Transfer:
    """
    Class transfer results of run between workspace folder and a capsule folder to cnetralize run results.
    """

    def __init__(self, workspace_path, runs_capsule_dir):
        self.workspace_path = workspace_path 
        self.runs_capsule_dir = runs_capsule_dir
        self.config_llm = LLMConfig.from_dict({"model": "deepseek-chat", "provider": "deepseek"})

    def create_capsule_name(self, goal: str) -> str:
        """
        Generate a unique lowercase folder name from goal using LLM.
        """
        system_prompt = """Generate a concise, unique lowercase folder name (max 7 words, underscore-separated) from the goal sentence. Only output the folder name, nothing else."""

        try:
            raw_output = LLMProvider(
                "capsule_namer", 
                system_msg=system_prompt, 
                config=self.config_llm
            )(f"generate a unique folder name for sentence: {goal}")
            name = raw_output.strip().lower()
            name = re.sub(r'^["\']|["\']$', '', name)  # Remove quotes
            name = re.sub(r'[^a-z0-9_]', '_', name)    # Sanitize
            name = re.sub(r'_+', '_', name)             # Collapse multiple underscores
            name = name.strip('_')                      # Remove leading/trailing
            if not name or len(name) < 5:
                raise ValueError("LLM output too short")
            if len(name) > 50:
                parts = name.split('_')[:7]
                name = '_'.join(parts)
            return name
        except Exception as e:
            raise e
    
    def create_capsule_folder(self, capsule_name):
        path = f"{self.runs_capsule_dir}/{capsule_name}"
        os.makedirs(path, exist_ok=True)
        return path
    
    def copy_files_recursive(self, source: Path, target: Path):
        """
        Recursively copy files and directories from source to target.

        Args:
            source: Source directory path
            target: Target directory path
        """
        if not source.exists():
            raise FileNotFoundError(f"Source directory '{source}' does not exist")

        if not source.is_dir():
            raise ValueError(f"Source '{source}' is not a directory")

        target.mkdir(parents=True, exist_ok=True)
        for item in source.iterdir():
            source_path = item
            target_path = target / item.name
            if source_path.is_dir():
                self.copy_files_recursive(source_path, target_path)
            else:
                try:
                    shutil.copy2(source_path, target_path)
                    print(f"Copied: {source_path} -> {target_path}")
                except Exception as e:
                    print(f"Error copying {source_path}: {e}")

    def transfer_workspace_files_to_capsule(self, goal):
        capsule_name = self.create_capsule_name(goal)
        path_capsule = Path(self.create_capsule_folder(capsule_name))
        path_workspace = Path(self.workspace_path)
        self.copy_files_recursive(path_workspace, path_capsule)
        self.clean_workspace()

    def clean_workspace(self):
        """Remove all files and directories in the workspace folder."""
        workspace = Path(self.workspace_path)
        shutil.rmtree(workspace, ignore_errors=True)
        workspace.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    trans = Transfer(workspace_path="/Users/cnrs/Documents/repository/toolomics/workspace", runs_capsule_dir="/Users/cnrs/Documents/repository/Mimosa-AI/runs_capsule")
    goal = "you are assigned Low-Light Image Enhancement with Wavelet-based Diffusion Models to replicate. You need to ACTUALLY EXECUTE the experiment only for the LOLv1 dataset. Paper is availableat this link https://arxiv.org/pdf/2306.00306. Reproduction without model training (inference only using available evaluation script is allowed). Looking for existing model weight if necessary is allowed. make sure plan is within json"
    trans.transfer_workspace_files_to_capsule(goal)
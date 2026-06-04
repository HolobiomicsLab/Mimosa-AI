
import shutil
from pathlib import Path
from typing import Union
from pathlib import Path
from sources.core.llm_provider import LLMProvider, LLMConfig, extract_model_pattern
import re
import os

class LocalTransfer:
    """
    Class transfer results of run between workspace folder and a capsule folder to cnetralize run results.
    """

    def __init__(self, config, workspace_path, runs_capsule_dir = "runs_capsule"):
        self.workspace_path = workspace_path
        self.runs_capsule_dir = runs_capsule_dir
        provider, model = extract_model_pattern(config.capsule_namer_model)
        # max_tokens kept small (256) — we only need a short folder name.
        self.config_llm = LLMConfig.from_dict({
            "model": model,
            "provider": provider,
            "max_tokens": 256,
        })

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

    def create_capsule_folder(self, capsule_name) -> str:
        path = f"{self.runs_capsule_dir}/{capsule_name}"
        os.makedirs(path, exist_ok=True)
        return path

    def count_files_recursive(self, directory: Path) -> int:
        """
        Count total number of files in a directory recursively.

        Args:
            directory: Directory path to count files in

        Returns:
            Total number of files (excluding directories)
        """
        if not directory.exists():
            return 0

        if not directory.is_dir():
            return 0

        count = 0
        try:
            for item in directory.rglob('*'):
                if item.is_file():
                    count += 1
        except Exception as e:
            print(f"Error counting files in {directory}: {e}")

        return count

    def copy_files_recursive(self, source: Path, target: Path) -> int:
        """
        Recursively copy files and directories from source to target.

        Args:
            source: Source directory path
            target: Target directory path

        Returns:
            Number of files successfully copied
        """
        if not source.exists():
            raise FileNotFoundError(f"Source directory '{source}' does not exist")

        if not source.is_dir():
            raise ValueError(f"Source '{source}' is not a directory")

        target.mkdir(parents=True, exist_ok=True)

        files_copied = 0
        for item in source.iterdir():
            source_path = item
            target_path = target / item.name
            if source_path.is_dir():
                files_copied += self.copy_files_recursive(source_path, target_path)
            else:
                try:
                    shutil.copy2(source_path, target_path)
                    files_copied += 1
                except Exception as e:
                    print(f"Error copying {source_path}: {e}")

        return files_copied

    def transfer_files_to_workspace(self, path_files_folder: str) -> int:
        """
        Transfer files from a source folder to the workspace.

        Args:
            path_files_folder: Path to source folder containing files to transfer

        Returns:
            Number of files successfully transferred

        Raises:
            FileNotFoundError: If source folder doesn't exist
            ValueError: If no files were transferred
        """
        folder_name = path_files_folder.split('/')[-1]
        path_destination = Path(f"{self.workspace_path}/{folder_name}")
        path_files = Path(path_files_folder)

        if not path_files.exists():
            raise FileNotFoundError(f"Source folder does not exist: {path_files}")

        if not path_files.is_dir():
            raise ValueError(f"Source path is not a directory: {path_files}")

        # Count files in source before transfer
        source_file_count = self.count_files_recursive(path_files)

        # Perform the transfer
        files_copied = self.copy_files_recursive(path_files, path_destination)

        # Verify files were actually transferred
        if files_copied == 0:
            raise ValueError(
                f"No files were transferred from {path_files} to {path_destination}. "
                f"Source contained {source_file_count} files."
            )

        # Verify workspace has files after transfer
        workspace_file_count = self.count_files_recursive(Path(self.workspace_path))
        if workspace_file_count == 0:
            raise ValueError(
                f"Workspace is empty after transfer. "
                f"Expected {files_copied} files but found none."
            )

        return files_copied

    def transfer_workspace_files_to_capsule(self, goal) -> str:
        capsule_name = self.create_capsule_name(goal)
        path_capsule = Path(self.create_capsule_folder(capsule_name))
        path_workspace = Path(self.workspace_path)
        self.copy_files_recursive(path_workspace, path_capsule)
        self.clean_workspace()
        return capsule_name

    def clean_workspace(self) -> None:
        """Remove all files and directories in the workspace folder."""
        workspace = Path(self.workspace_path)
        if workspace.exists() and workspace.is_dir():
            for item in workspace.iterdir():
                if item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
                else:
                    item.unlink(missing_ok=True)

if __name__ == "__main__":
    trans = LocalTransfer(workspace_path="/Users/cnrs/Documents/repository/toolomics/workspace", runs_capsule_dir="/Users/cnrs/Documents/repository/Mimosa-AI/runs_capsule")
    goal = "you are assigned Low-Light Image Enhancement with Wavelet-based Diffusion Models to replicate. You need to ACTUALLY EXECUTE the experiment only for the LOLv1 dataset. Paper is availableat this link https://arxiv.org/pdf/2306.00306. Reproduction without model training (inference only using available evaluation script is allowed). Looking for existing model weight if necessary is allowed. make sure plan is within json"
    trans.transfer_workspace_files_to_capsule(goal)

import os

def list_files(path='.', max_depth=2):
    """
    Always returns paths relative to the current working directory.
    """
    files = []
    base_path = os.path.abspath(path)
    
    for root, _, filenames in os.walk(base_path):
        relative_root = os.path.relpath(root, base_path)
        current_depth = 0 if relative_root == '.' else relative_root.count(os.sep) + 1
        if current_depth > max_depth:
            continue
            
        for filename in filenames:
            file_path = os.path.join(root, filename)
            rel_path = os.path.relpath(file_path, base_path)
            files.append(rel_path)
    return '\n'.join(files)
import subprocess

def runner(code: str) -> str:
    print("\n🔧 Executing generated workflow in sandbox...")
    process = subprocess.Popen(
        ["python", "-c", code], 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    print("\n📊 Execution Progress:")
    for line in iter(process.stdout.readline, ''):
        print(line, end='')
    stderr_output = process.stderr.read()
    process.wait()
    print("\nExited.\n🔍 Errors (if any):")
    print(stderr_output)
    return process.returncode
import subprocess

def run_python_command_in_venv(venv_python_path, command):
    """
    Runs a Python command using the Python interpreter from a virtual environment.

    Args:
    venv_python_path (str): The path to the Python interpreter in the virtual environment.
    command (str): The Python command to execute.
    """
    try:
        result = subprocess.run([venv_python_path, '-c', command], check=True, capture_output=True, text=True)
        print("Output:", result.stdout)
        print("Error:", result.stderr)
    except subprocess.CalledProcessError as e:
        print("Error during command execution:", e)

# Define the path to the venv Python interpreter
venv_python_path = '/usr/local/envs/venv/bin/python'

# Command to install requirements from requirements.txt
command = 'import subprocess; subprocess.run(["python", "--version"], check=True)'

# Run the command in the virtual environment
run_python_command_in_venv(venv_python_path, command)

# Command to install requirements from requirements.txt
command = 'import subprocess; subprocess.run(["pip", "install", "-r", "/content/inswapper/requirements.txt"], check=True)'
run_python_command_in_venv(venv_python_path, command)

# Command to install requirements from requirements.txt
command = 'import subprocess; subprocess.run(["pip", "install", "git+https://github.com/sajjjadayobi/FaceLib.git"], check=True)'
run_python_command_in_venv(venv_python_path, command)

import pkg_resources

# Check the installed version of FaceLib
try:
    version = pkg_resources.get_distribution('facelib').version
    print(f"FaceLib version: {version}")
except pkg_resources.DistributionNotFound:
    print("FaceLib is not installed.")

try:
    from facelib.utils import face_restoration_helper
    print("Module face_restoration_helper is available in FaceLib.")
except ImportError as e:
    print("Module face_restoration_helper not found in FaceLib:", e)

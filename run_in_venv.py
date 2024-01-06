import subprocess
import site
import os

# Define the path to the venv Python interpreter
venv_python_path = '/usr/local/envs/venv/bin/python'

# Function to run pip command in the virtual environment
def run_pip_command_in_venv(venv_python_path, pip_command):
    """
    Runs a pip command using the Python interpreter from a virtual environment.
    Args:
    venv_python_path (str): The path to the Python interpreter in the virtual environment.
    pip_command (list): The pip command to execute as a list.
    """
    try:
        # Include the Python interpreter path in the command
        command = [venv_python_path, '-m'] + pip_command
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("Output:", result.stdout)
        print("Error:", result.stderr)
    except subprocess.CalledProcessError as e:
        print("Error during command execution:", e)

# Run the command in the virtual environment to install packages from requirements.txt
# run_pip_command_in_venv(venv_python_path, ['pip', 'install', '-r', '/content/inswapper/requirements.txt'])

# Install facelib
# run_pip_command_in_venv(venv_python_path, ['pip', 'install', 'git+https://github.com/sajjjadayobi/FaceLib.git'])
# run_pip_command_in_venv(venv_python_path, ['pip', 'install', 'git+https://github.com/xinntao/facexlib.git'])

# Now the packages should be installed, and you can check if facelib was installed by running
# `conda list` in the terminal or restarting the notebook kernel and importing facelib.

# List all site-packages directories
site_packages_dirs = site.getsitepackages()

# Look for 'facelib' in each directory
for dir in site_packages_dirs:
    potential_path = os.path.join(dir, 'facelib')
    if os.path.exists(potential_path):
        print("Facelib found in:", potential_path)
        break
else:
    print("Facelib not found in standard site-packages directories.")


utils_path = os.path.join(potential_path, 'utils')
if os.path.exists(utils_path):
    print("Contents of 'utils':", os.listdir(utils_path))
else:
    print("'utils' directory not found in facelib.")

helper_path = os.path.join(utils_path, 'face_restoration_helper.py')
if os.path.exists(helper_path):
    print("'face_restoration_helper.py' found in 'utils'.")
else:
    print("'face_restoration_helper.py' not found in 'utils'.")




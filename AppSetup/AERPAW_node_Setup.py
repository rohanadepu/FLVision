#########################################################
#    Imports and env setup                               #
#########################################################
import os
import subprocess
import sys


#########################################################
#    Automate Functions                               #
#########################################################

# Function to install system dependencies
def install_system_dependencies():
    try:
        print("Updating package list...")
        subprocess.check_call(["sudo", "apt", "update"])

        print("Installing system dependencies...")
        packages = ["python3-pip", "libcairo2-dev", "pkg-config", "python3-dev", "unzip"]
        subprocess.check_call(["sudo", "apt", "install", "-y"] + packages)

        print("System dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error installing system dependencies: {e}")
    except FileNotFoundError as e:
        print(f"Error: apt command not found. Make sure you're running on a Debian/Ubuntu system: {e}")


# Function to install dependencies from requirements_edge.txt
def install_python_dependencies(requirements_file):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
        print(f"Dependencies installed from {requirements_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
    except FileNotFoundError as e:
        print(f"Requirements file not found: {e}")


# Function to create folder structure
def create_folder_structure(base_dir):
    try:
        root_dir = "datasets"
        sub_dirs = ["IOTBOTNET2020", "CICIOT2023", "LIVEDATA"]

        # Create root directories
        root_path = os.path.join(base_dir, root_dir)
        os.makedirs(root_path, exist_ok=True)

        for sub_dir in sub_dirs:
            # Create subdirectories
            sub_path = os.path.join(root_path, sub_dir)
            os.makedirs(sub_path, exist_ok=True)

        print(f"Folder structure created in {base_dir}")
    except Exception as e:
        print(f"Error creating folder structure: {e}")


# Function to clone a Git repository
def clone_git_repo(repo_url, clone_dir):
    try:
        subprocess.check_call(["git", "clone", repo_url, clone_dir])
        print(f"Repository cloned into {clone_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error cloning repository: {e}")


#########################################################
#    Execution                                          #
#########################################################

if __name__ == "__main__":
    # --- Settings for AERPAW (change paths for Chameleon) --- #

    base_directory = "/root"  # For Chameleon: "/home/cc"
    requirements_file_path = "/root/HFL-DNN-GAN-IDS/AppSetup/requirements_edge.txt"  # For Chameleon: "/home/cc/HFL-DNN-GAN-IDS/AppSetup/requirements_core.txt"
    git_repo_url = "https://github.com/keko787/HFL-DNN-GAN-IDS.git"
    clone_dir = "/root/HFL-DNN-GAN-IDS"  # For Chameleon: "/home/cc/HFL-DNN-GAN-IDS"

    # --- Steps --- #

    # First, install system dependencies
    install_system_dependencies()

    # Second, Create the folder structure for the datasets
    create_folder_structure(base_directory)

    # Third, clone the project into the node
    # clone_git_repo(git_repo_url, clone_dir)

    # Fourth, install the Python dependencies from the project
    install_python_dependencies(requirements_file_path)
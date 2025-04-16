#########################################################
#    Imports and env setup                               #
#########################################################
import os
import subprocess
import sys

#########################################################
#    Automate Functions                               #
#########################################################

# Function to create folder structure
def create_folder_structure(base_dir):
    try:
        root_dir = "datasets"
        sub_dirs = ["IOTBOTNET2020", "CICIOT2023"]

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


# Function to install dependencies from requirements_edge.txt
def install_dependencies(requirements_file):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
        print(f"Dependencies installed from {requirements_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
    except FileNotFoundError as e:
        print(f"Requirements file not found: {e}")


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

    # --- Settings --- #

    base_directory = "/home/cc"  # Replace with your base directory path
    requirements_file_path = "/home/cc/HFL-DNN-GAN-IDS/SetUp_Scripts/requirements_core.txt"  # Replace with the path to your requirements_edge.txt
    git_repo_url = "https://github.com/keko787/HFL-DNN-GAN-IDS.git"  # Replace with your Git repository URL
    clone_dir = "/home/cc/HFL-DNN-GAN-IDS"

    # --- Steps --- #

    # First, Create the folder structure for the datasets
    create_folder_structure(base_directory)

    # Second, clone the project into the node
    # clone_git_repo(git_repo_url, clone_dir)

    # third, install the dependencies from the project
    install_dependencies(requirements_file_path)


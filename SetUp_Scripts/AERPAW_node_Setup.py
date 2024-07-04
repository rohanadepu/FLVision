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
        root_dirs = ["trainingDataset", "poisonedDataset"]
        sub_dirs = ["iotbotnet2020"]
        sub_sub_dirs = ["ddos", "dos", "scan", "theft"]
        ddos_dirs = ["ddos_http", "ddos_udp", "ddos_tcp"]
        dos_dirs = ["dos_http", "dos_udp", "dos_tcp"]
        scan_dirs = ["os", "service"]
        theft_dirs = ["data_exfiltration", "keylogging"]

        for root_dir in root_dirs:
            # Create root directories
            root_path = os.path.join(base_dir, root_dir)
            os.makedirs(root_path, exist_ok=True)

            for sub_dir in sub_dirs:
                # Create subdirectories
                sub_path = os.path.join(root_path, sub_dir)
                os.makedirs(sub_path, exist_ok=True)

                for sub_sub_dir in sub_sub_dirs:
                    sub_sub_path = os.path.join(sub_path, sub_sub_dir)
                    os.makedirs(sub_sub_path, exist_ok=True)

                    if sub_sub_dir == "ddos":
                        for ddos_dir in ddos_dirs:
                            os.makedirs(os.path.join(sub_sub_path, ddos_dir), exist_ok=True)
                    elif sub_sub_dir == "dos":
                        for dos_dir in dos_dirs:
                            os.makedirs(os.path.join(sub_sub_path, dos_dir), exist_ok=True)
                    elif sub_sub_dir == "scan":
                        for scan_dir in scan_dirs:
                            os.makedirs(os.path.join(sub_sub_path, scan_dir), exist_ok=True)
                    elif sub_sub_dir == "theft":
                        for theft_dir in theft_dirs:
                            os.makedirs(os.path.join(sub_sub_path, theft_dir), exist_ok=True)

        print(f"Folder structure created in {base_dir}")
    except Exception as e:
        print(f"Error creating folder structure: {e}")


# Function to install dependencies from requirements.txt
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

    base_directory = "/root"  # Replace with your base directory path
    requirements_file_path = "/root/FLVision/SetUp_Scripts/requirements.txt"  # Replace with the path to your requirements.txt
    git_repo_url = "https://github.com/rohanadepu/FLVision.git"  # Replace with your Git repository URL
    clone_dir = "/root/FLVision"

    # --- Steps --- #

    # First, Create the folder structure for the datasets
    create_folder_structure(base_directory)

    # Second, clone the project into the node
    clone_git_repo(git_repo_url, clone_dir)

    # third, install the dependencies from the project
    install_dependencies(requirements_file_path)


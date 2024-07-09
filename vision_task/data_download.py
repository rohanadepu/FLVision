import os
import gdown
import zipfile

# Define paths for the datasets
data_dir = os.path.join(os.getcwd(), 'data', 'VisDrone')
train_zip_path = os.path.join(data_dir, 'VisDrone2019-DET-train.zip')
test_zip_path = os.path.join(data_dir, 'VisDrone2019-DET-test-dev.zip')

# Google Drive links (formatted for gdown)
train_url = 'https://drive.google.com/uc?id=1a2oHjcEcwXP8oUF95qiwrqzACb2YlUhn'
test_url = 'https://drive.google.com/uc?id=1PFdW_VFSCfZ_sTSZAGjQdifF_Xd5mf0V'

# Create data directory if it doesn't exist
os.makedirs(data_dir, exist_ok=True)

# Function to download datasets
def download_dataset(url, zip_path):
    if not os.path.exists(zip_path):
        print(f"Downloading {zip_path}...")
        gdown.download(url, zip_path, quiet=False)
    else:
        print(f"{zip_path} already exists. Skipping download.")

# Function to extract datasets
def extract_dataset(zip_path, sub_dir):
    if os.path.exists(zip_path):
        extract_dir = os.path.join(data_dir, sub_dir)
        os.makedirs(extract_dir, exist_ok=True)  # Create the directory if it does not exist
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"Extracted to {extract_dir}")

# Download datasets
download_dataset(train_url, train_zip_path)
download_dataset(test_url, test_zip_path)

# Extract datasets
extract_dataset(train_zip_path, 'train')  
extract_dataset(test_zip_path, 'test') 

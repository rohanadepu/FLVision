import os
import pandas as pd
import random



# Function to load all files from a given directory
def load_files_from_directory(directory, file_extension=".csv", sample_size=None):
    dataframes = []
    all_files = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(file_extension):
                file_path = os.path.join(root, file)
                all_files.append(file_path)

    # Shuffle and sample files if sample_size is specified
    if sample_size is not None:
        random.shuffle(all_files)
        all_files = all_files[:sample_size]
        print(all_files)

    for file_path in all_files:
        df = pd.read_csv(file_path)  # Modify this line if files are in a different format
        dataframes.append(df)

    return dataframes

sample_size = 2

# Load DDoS UDP files
ddos_udp_directory = '../iotbotnet2020_archive/ddos/DDOS UDP'
ddos_udp_dataframes = load_files_from_directory(ddos_udp_directory, sample_size=sample_size)

# Load DDoS TCP files
ddos_tcp_directory = '../iotbotnet2020_archive/ddos/DDOS TCP'
ddos_tcp_dataframes = load_files_from_directory(ddos_tcp_directory, sample_size=sample_size)

# Load DDoS HTTP files
ddos_http_directory = '../iotbotnet2020_archive/ddos/DDOS HTTP'
ddos_http_dataframes = load_files_from_directory(ddos_http_directory)

# Load DoS UDP files
dos_udp_directory = '../iotbotnet2020_archive/dos/dos udp'
dos_udp_dataframes = load_files_from_directory(dos_udp_directory, sample_size=sample_size)

# Load DDoS TCP files
dos_tcp_directory = '../iotbotnet2020_archive/dos/dos tcp'
dos_tcp_dataframes = load_files_from_directory(dos_tcp_directory, sample_size=sample_size)

# Load DDoS HTTP files
dos_http_directory = '../iotbotnet2020_archive/dos/dos_http'
dos_http_dataframes = load_files_from_directory(dos_http_directory)

# Load scan_os files
scan_os_directory = '../iotbotnet2020_archive/scan/os'
scan_os_dataframes = load_files_from_directory(scan_os_directory)

# Load scan_service files
scan_service_directory = '../iotbotnet2020_archive/scan/service'
scan_service_dataframes = load_files_from_directory(scan_service_directory)

# Load theft_data_exfiltration files
theft_data_exfiltration_directory = '../iotbotnet2020_archive/theft/data_exfiltration'
theft_data_exfiltration_dataframes = load_files_from_directory(theft_data_exfiltration_directory)

# Load theft_keylogging files
theft_keylogging_directory = '../iotbotnet2020_archive/theft/keylogging'
theft_keylogging_dataframes = load_files_from_directory(theft_keylogging_directory)

# Optionally, concatenate all dataframes if needed
all_ddos_udp_data = pd.concat(ddos_udp_dataframes, ignore_index=True)
all_ddos_tcp_data = pd.concat(ddos_tcp_dataframes, ignore_index=True)
all_ddos_http_data = pd.concat(ddos_http_dataframes, ignore_index=True)
all_dos_udp_data = pd.concat(dos_udp_dataframes, ignore_index=True)
all_dos_tcp_data = pd.concat(dos_tcp_dataframes, ignore_index=True)
all_dos_http_data = pd.concat(dos_http_dataframes, ignore_index=True)
all_scan_os_data = pd.concat(scan_os_dataframes, ignore_index=True)
all_scan_service_data = pd.concat(scan_service_dataframes, ignore_index=True)
all_theft_data_exfiltration_data = pd.concat(theft_data_exfiltration_dataframes, ignore_index=True)
all_theft_keylogging_data = pd.concat(theft_keylogging_dataframes, ignore_index=True)

# Display the first few rows of the combined DataFrames
print("DDoS UDP Data:")
print(all_ddos_udp_data.head())

print("DDoS TCP Data:")
print(all_ddos_tcp_data.head())

print("DDoS HTTP Data:")
print(all_ddos_http_data.head())

print("DoS UDP Data:")
print(all_dos_udp_data.head())

print("DoS TCP Data:")
print(all_dos_tcp_data.head())

print("DoS HTTP Data:")
print(all_dos_http_data.head())

print("Scan OS Data:")
print(all_scan_os_data.head())

print("Scan Service Data:")
print(all_scan_service_data.head())

print("Theft Data Exfiltration Data:")
print(all_theft_data_exfiltration_data.head())

print("Theft Keylogging Data:")
print(all_theft_keylogging_data.head())

# Save the combined DataFrames to new CSV files if needed
# all_ddos_udp_data.to_csv('combined_ddos_udp_data.csv', index=False)
# all_dos_udp_data.to_csv('combined_dos_udp_data.csv', index=False)
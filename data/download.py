import os

from azure.storage.blob import ContainerClient

# Azure storage account details for the open dataset
account_name = "azureopendatastorage"
container_name = "nyctlc"
folder_name = "yellow"

# Construct the account URL
account_url = f"https://{account_name}.blob.core.windows.net"

# Create the ContainerClient with the correct parameters
container_client = ContainerClient(
    account_url=account_url, container_name=container_name, credential=None
)

# Local directory to store the downloaded files
local_dir = "nyc_yellow_taxi_parquet"
if not os.path.exists(local_dir):
    os.makedirs(local_dir)

print("Listing blobs in the 'yellow' folder...")
# List all blobs that start with the folder name
blobs_list = container_client.list_blobs(name_starts_with=folder_name)

for blob in blobs_list:
    blob_name = blob.name
    if blob_name.endswith(".parquet"):
        print(f"Downloading {blob_name} ...")
        blob_client = container_client.get_blob_client(blob_name)
        # Save only the file name (without the folder path)
        local_file_path = os.path.join(local_dir, os.path.basename(blob_name))
        with open(local_file_path, "wb") as f:
            download_stream = blob_client.download_blob()
            f.write(download_stream.readall())
        print(f"Downloaded {local_file_path}")

print("Download complete!")

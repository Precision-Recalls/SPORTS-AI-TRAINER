from azure.storage.blob import BlobServiceClient
from common.utils import load_config

config = load_config('configs/config.ini')
connection_string = config['azure']['connection_string']
container_name = config['azure']['container_name']

blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_client = blob_service_client.get_container_client(container_name)

def upload_blob(blob_name, data):
    blob_client = container_client.get_blob_client(blob_name)
    blob_client.upload_blob(data, overwrite=True)

def download_blob(blob_name):
    blob_client = container_client.get_blob_client(blob_name)
    return blob_client.download_blob().readall()

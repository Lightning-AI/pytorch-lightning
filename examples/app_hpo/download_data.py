from utils import download_data

data_dir = "hymenoptera_data_version_0"
download_url = f"https://pl-flash-data.s3.amazonaws.com/{data_dir}.zip"
download_data(download_url, "./data")

import lightning_app
from lightning_app.utilities.packaging.lightning_utils import download_frontend

if __name__ == "__main__":
    download_frontend(lightning_app._PROJECT_ROOT)

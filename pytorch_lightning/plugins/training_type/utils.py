import os


def on_colab_kaggle() -> bool:
    return bool(os.getenv("COLAB_GPU") or os.getenv("KAGGLE_URL_BASE"))

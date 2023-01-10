import os
from dataclasses import dataclass


@dataclass
class _Config:
    id = os.getenv("LIGHTNING_USER_ID")
    key = os.getenv("LIGHTNING_API_KEY")
    url = os.getenv("LIGHTNING_CLOUD_URL", "https://lightning.ai")
    api_key = os.getenv("LIGHTNING_API_KEY")
    username = os.getenv("LIGHTNING_USERNAME")
    video_location = os.getenv("VIDEO_LOCATION", "./artifacts/videos")
    har_location = os.getenv("HAR_LOCATION", "./artifacts/hars")
    slowmo = os.getenv("SLOW_MO", "0")

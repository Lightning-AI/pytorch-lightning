from time import sleep

import requests
from lightning.app.utilities.state import headers_for

headers = headers_for({})
headers["X-Lightning-Type"] = "DEFAULT"

res = requests.get("http://127.0.0.1:7501/state", headers=headers)


res = requests.post("http://127.0.0.1:7501/state", json={"stage": "running"}, headers=headers)
print(res)

sleep(10)

res = requests.post("http://127.0.0.1:7501/state", json={"stage": "stopping"}, headers=headers)
print(res)

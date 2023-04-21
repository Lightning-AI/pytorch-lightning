# app.py
import lightning as L


class YourComponent(L.LightningWork):
    def run(self):
        print("RUN ANY PYTHON CODE HERE")


if __name__ == "__main__":
    component = YourComponent()
    app = L.LightningApp(component)

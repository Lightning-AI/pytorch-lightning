from pathlib import Path

from lightning.app.components.python import PopenPythonScript

if __name__ == "__main__":
    comp = PopenPythonScript(Path(__file__).parent / "pl_script.py")
    comp.run()

import os
import sys

from lightning.app import LightningApp

if __name__ == "__main__":
    sys.path.append(os.path.dirname(__file__))
    from comp_dockerfile.a.a import AAA
    from comp_dockerfile.b.b import BBB

    app = LightningApp(BBB(AAA()))

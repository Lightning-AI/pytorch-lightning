import os
import sys

from lightning.app import LightningApp

if __name__ == "__main__":
    sys.path.append(os.path.dirname(__file__))

    from comp_req.a.a import A
    from comp_req.b.b import B

    app = LightningApp(B(A()))

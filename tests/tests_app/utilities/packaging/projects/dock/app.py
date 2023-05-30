import os
import sys

from lightning.app import LightningApp

if __name__ == "__main__":
    sys.path.append(os.path.dirname(__file__))

    from compo.a.a import AA
    from compo.b.b import BB

    app = LightningApp(BB(AA()))

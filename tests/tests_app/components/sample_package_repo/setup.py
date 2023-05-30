import json
import os

from setuptools import find_packages, setup
from setuptools.command.install import install

LIGHTNING_COMPONENT_INFO = {
    "package": "external_lightning_component_package",
    "version": "0.0.1",
    "entry_point": "myorg.lightning_modules",
}


class PostInstallCommand(install):
    def run(self):
        install.run(self)
        os.system(f"echo Installed lightning component package: {json.dumps(json.dumps(LIGHTNING_COMPONENT_INFO))}")


setup(
    name=LIGHTNING_COMPONENT_INFO["package"],
    version=LIGHTNING_COMPONENT_INFO["version"],
    description="example of an external lightning package that contains lightning components",
    author="manskx",
    author_email="mansy@grid.ai",
    url="grid.ai",
    download_url="https://github.com/Lightning-AI/lightning",
    license="TBD",
    packages=find_packages(exclude=["tests", "docs"]),
    long_description="example of an external lightning package that contains lightning components",
    long_description_content_type="text/markdown",
    include_package_data=True,
    zip_safe=False,
    keywords=["deep learning", "pytorch", "AI"],
    python_requires=">=3.6",
    entry_points={
        "lightning.app.external_components": [
            f"{LIGHTNING_COMPONENT_INFO['entry_point']}= "
            f"{LIGHTNING_COMPONENT_INFO['package']}:exported_lightning_components",
        ],
    },
    cmdclass={
        "install": PostInstallCommand,
    },
    setup_requires=["wheel"],
)

################################################################
How to run an app on the cloud with a local version of lightning
################################################################

The lightning cloud uses the latest release by default. However, you might want to run your app with some local changes you've made to the lightning framework. To use your local version of lightning on the cloud, set the following environment variable:

```bash
git clone https://github.com/Lightning-AI/lightning.git
cd lightning
pip install -e .
export PACKAGE_LIGHTNING=1  # <- this is the magic to use your version (not mainstream PyPI)!
lightning_app run app app.py --cloud
```

By setting `PACKAGE_LIGHTNING=1`, lightning packages the lightning source code in your local directory in addition to your app source code and uploads them to the cloud.

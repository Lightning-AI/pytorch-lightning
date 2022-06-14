from six.moves import urllib

# TorchVision hotfix https://github.com/pytorch/vision/issues/1938
opener = urllib.request.build_opener()
opener.addheaders = [("User-agent", "Mozilla/5.0")]
urllib.request.install_opener(opener)

from lightning_fabric import Fabric
from lightning_fabric.plugins.precision import BitsandbytesQuantization

quantization_plugin = BitsandbytesQuantization(mode="bnb.int8")

fabric = Fabric(plugins=quantization_plugin)

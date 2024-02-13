import numpy as np
from lightning.data import optimize
from PIL import Image


# Write random images into the chunks
def random_images(index):
  return {
    "index": index,
    "image": Image.fromarray(np.random.randint(0, 256, (32, 32, 3), np.uint8)),
    "class": np.random.randint(10),
  }

if __name__ == "__main__":
    optimize(
        fn=random_images,  # The function applied over each input.
        inputs=list(range(1000)),  # any inputs. This is provided to your function.
        output_dir="my_dataset",  # where to store the optimized data.
        compression="zstd",  # apply any compression.
        num_workers=4,  # Distribute the inputs across multiple workers.
        chunk_bytes="64MB"  # The maximum number of bytes to write into a chunk.
    )

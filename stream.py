from lightning.data import StreamingDataset
from torch.utils.data import DataLoader

# Remote path where full dataset is persistently stored
input_dir = 's3://pl-flash-data/my_dataset'

# Create streaming dataset
dataset = StreamingDataset(input_dir, shuffle=True)

# Let's see what is in sample #1337...
sample = dataset[50]
img = sample['image']
cls = sample['class']

# Create PyTorch DataLoader
dataloader = DataLoader(dataset)

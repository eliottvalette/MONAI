from monai.transforms import LoadImaged
from monai.data import Dataset, DataLoader
import os

# Define the image path
image_path = "tests/testing_data/anatomical_label.nii.gz"

# Create a sample data dictionary
data = [{"image": image_path}]

# Define transforms with image_only=True
transform_image_only = LoadImaged(keys=["image"], image_only=True, reader="NibabelReader")

# Define transforms with image_only=False
transform_with_meta = LoadImaged(keys=["image"], image_only=False, reader="NibabelReader")

# Create datasets
dataset_image_only = Dataset(data=data, transform=transform_image_only)
dataset_with_meta = Dataset(data=data, transform=transform_with_meta)

# Create data loaders
loader_image_only = DataLoader(dataset_image_only, batch_size=1)
loader_with_meta = DataLoader(dataset_with_meta, batch_size=1)

# Get the first batch from each loader
batch_image_only = next(iter(loader_image_only))
batch_with_meta = next(iter(loader_with_meta))

# Print the outputs
print("Output with image_only=True:")
print(batch_image_only)
print("\nOutput with image_only=False:")
print(batch_with_meta)
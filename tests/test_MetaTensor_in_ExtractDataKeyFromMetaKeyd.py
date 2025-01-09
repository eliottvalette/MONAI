import json
from monai.transforms.transform import MapTransform, Randomizable
from monai.apps.reconstruction.transforms.dictionary import ExtractDataKeyFromMetaKeyd
import torch
from monai.data import MetaTensor
import torch

'''
# Simulated input data
data = {
    "image": torch.randn(1, 5, 5),  # Image tensor
    "meta": {  # Metadata dictionary
        "reconstruction_rss": torch.ones(5, 5),  # Ground truth image stored in meta
        "patient_id": "12345"
    }
}

# Define the transform
extract_key_transform = ExtractDataKeyFromMetaKeyd(
    keys=["reconstruction_rss", "patient_id"],  # Keys to move from meta to data
    meta_key="meta",  # Where the metadata is stored
    allow_missing_keys=False  # Raise an error if key is missing
)

# Apply the transform
result = extract_key_transform(data)

# Output the transformed data in pretty JSON format
print("Transformed Data:")
with open('first-issue-tests.json', 'w') as f:
    f.write(json.dumps(result, indent=4, default=str))
'''

# Create a standard PyTorch tensor
data = torch.tensor([[1, 2], [3, 4]])

# Create an affine matrix for spatial transformation (optional)
affine = torch.tensor(
    [[2, 0, 0, 0],
     [0, 2, 0, 0],
     [0, 0, 2, 0],
     [0, 0, 0, 1]], dtype=torch.float64
)

# Add metadata to the tensor
meta = {"description": "This is a sample tensor", "patient_id": "12345"}

# Create a MetaTensor
meta_tensor = MetaTensor(data, affine=affine, meta=meta)
print(meta_tensor)

# Apply ExtractDataKeyFromMetaKeyd to a MetaTensor object
extract_key_transform = ExtractDataKeyFromMetaKeyd(
    keys=["description", "patient_id"],  # Keys to move from meta to data
    meta_key="meta",  # Where the metadata is stored
    allow_missing_keys=False  # Raise an error if key is missing
)

print(extract_key_transform)
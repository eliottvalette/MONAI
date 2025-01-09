import json
import torch
from monai.apps.reconstruction.transforms.dictionary import ExtractDataKeyFromMetaKeyd
from monai.data import MetaTensor
import numpy as np

# Sample data for testing
data_mapping = {
    "image": "sample_image",
    "meta": {
        "reconstruction_rss": "sample_value1"
    }
}

affine_1 = torch.tensor(
    [[2, 0, 0, 0],
     [0, 2, 0, 0],
     [0, 0, 2, 0],
     [0, 0, 0, 1]], dtype=torch.float64
)

affine_2 = torch.tensor(
    [[3, 0, 0, 0],
     [0, 3, 0, 0],
     [0, 0, 3, 0],
     [0, 0, 0, 1]], dtype=torch.float64
)

data_meta_tensor_inplace = MetaTensor(
    np.array([[1, 2], [3, 4]]),
    affine=affine_1,
    meta={
        "meta": {"reconstruction_rss": "sample_value2", 
                   "patient_id": "sample_patient_id"
                },
        "not-meta": {"random_key": "random_value",
                     "random_key2": "random_value2"
                    }
        },
)

data_meta_tensor_not_inplace = MetaTensor(
    np.array([[5, 6], [8, 9]]),
    affine=affine_2,
    meta={"meta": {"reconstruction_rss": "sample_value3", 
                    "patient_id": "sample_patient_id"
                },
          "not-meta": {"random_key": "random_value",
                            "random_key2": "random_value2"
                }
           },
)

# Instantiate the class
extract_key_transform = ExtractDataKeyFromMetaKeyd(keys=["reconstruction_rss"], meta_key="meta", allow_missing_keys=False)

# Apply the transform for Mapping
print("Result for Mapping:")
result_mapping = extract_key_transform(data_mapping)
print(json.dumps(result_mapping, indent=4, default=str))

# Apply the transform for MetaTensor inplace
print("\nResult for MetaTensor inplace:")
result_meta_tensor_inplace = extract_key_transform(data_meta_tensor_inplace)
print('Inplace modified :', json.dumps(result_meta_tensor_inplace.meta, indent=4, default=str))
print('\n Output', json.dumps(result_meta_tensor_inplace.meta, indent=4, default=str))

# Apply the transform for MetaTensor not inplace
print("\nResult for MetaTensor not inplace:")
extract_key_transform.inplace = False
result_meta_tensor_not_inplace = extract_key_transform(data_meta_tensor_not_inplace)
print('Not Inplace Not modified :', json.dumps(data_meta_tensor_not_inplace.meta, indent=4, default=str))
print('\n Output', json.dumps(result_meta_tensor_not_inplace.meta, indent=4, default=str))
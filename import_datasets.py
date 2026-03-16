import json

import numpy as np
import os

dataset_path = "data/datasets"
destination_path = "ts2vec/datasets"

dataset_names = []

for filename in os.listdir(dataset_path):
    if filename.endswith(".npz"):
        file_path = os.path.join(dataset_path, filename)
        # Process the dataset file
        data = np.load(file_path)
        X = data["X"]
        X = X[:, :, None]
        print(f"X shape: {X.shape}")
        dataset_name = filename[:-4]
        dataset_names.append(dataset_name)
        save_path = os.path.join(destination_path, dataset_name + ".npy")
        np.save(save_path, X)

with open("ts2vec/datasets/dataset_names.json", "w") as f:
    json.dump(dataset_names, f)

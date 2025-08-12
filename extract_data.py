import joblib
import numpy as np
import os

# Load the original dataset
data_path = '/home/baai/Documents/asap-retarget/merged_classes/jump_movement_smoothnew_smoothExpand1.5s.pkl'
data = joblib.load(data_path)

# Create output directory if it doesn't exist
output_dir = 'poses'
os.makedirs(output_dir, exist_ok=True)

# Specify the key we want to extract
target_keys = ['0-CMU_91_91_37_poses', "0-MPI_mosh_00058_jiggles 2_poses", "0-BMLmovi_Subject_76_F_MoSh_Subject_76_F_1_poses", "0-BMLmovi_Subject_6_F_MoSh_Subject_6_F_1_poses", "0-BMLmovi_Subject_42_F_MoSh_Subject_42_F_17_poses"]

# Extract the specific motion data with all its components
for target_key in target_keys:
    new_data = {}
    if target_key in data:
        # Copy the motion data
        new_data[target_key] = data[target_key]
        
        # Save the new dataset
        output_filename = os.path.join(output_dir, f"{target_key}.pkl")
        joblib.dump(new_data, output_filename)
        
        print(f"Successfully saved dataset to {output_filename}")
        print(f"Dataset structure:")
        print(f"- Keys: {list(new_data.keys())}")
        print(f"- Motion data components:")
        for key, value in new_data[target_key].items():
            if isinstance(value, np.ndarray):
                print(f"  - {key}: shape {value.shape}")
            else:
                print(f"  - {key}: {value}")
    else:
        print(f"Key '{target_key}' not found in the dataset")
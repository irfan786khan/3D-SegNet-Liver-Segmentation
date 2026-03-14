import os
import shutil

# Define source and target directories
source_base_dir = r"E:\linkdoc\Code_work\9Samples\testData"
target_base_dir = r"E:\linkdoc\Code_work\9Samples\testData\watershedSegmentationOrderLiver_3d"

# Create the target directory if it doesn't exist
os.makedirs(target_base_dir, exist_ok=True)

# Loop through patient folders
for patient_id in os.listdir(source_base_dir):
    patient_folder = os.path.join(source_base_dir, patient_id)
    
    if os.path.isdir(patient_folder):
        # Look for any file containing 'liver_3d' and ending with .nii.gz
        for filename in os.listdir(patient_folder):
            if "liver_3d" in filename and filename.endswith(".nii.gz"):
                source_file = os.path.join(patient_folder, filename)
                target_file = os.path.join(target_base_dir, f"{patient_id}.nii.gz")
                shutil.copy2(source_file, target_file)
                print(f"Copied: {filename} -> {patient_id}.nii.gz")
                break  # Stop after first match
        else:
            print(f"No liver_3d file found in {patient_id}")

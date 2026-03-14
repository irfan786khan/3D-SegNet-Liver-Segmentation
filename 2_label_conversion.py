import os
import re
import nibabel as nib
import numpy as np
from tqdm import tqdm

# === Change these depending on train/test ===
source_root = r"E:\linkdoc\Code_work\9Samples\testData"
dst_root = r"E:\linkdoc\Code_work\9Samples\testData\testDataLabelsTr_hepatic_1_8"

# Create output folder
os.makedirs(dst_root, exist_ok=True)

# Regex: match hepatic/artery/vein with a number
# Example: hepatic3_3d.nii.gz, artery0_3d.nii.gz, vein2_001.nii.gz
pattern = re.compile(r"(hepatic|artery|vein)(\d+)")

for patient_id in tqdm(os.listdir(source_root)):
    patient_folder = os.path.join(source_root, patient_id)
    if not os.path.isdir(patient_folder):
        continue

    label_volume = None
    affine = None

    for fname in os.listdir(patient_folder):
        if not fname.endswith(".nii.gz"):
            continue

        match = pattern.search(fname)
        if match:
            structure = match.group(1)  # hepatic / artery / vein
            idx = int(match.group(2))   # the number after name

            fpath = os.path.join(patient_folder, fname)
            img = nib.load(fpath)
            data = img.get_fdata()
            affine = img.affine

            if label_volume is None:
                label_volume = np.zeros_like(data, dtype=np.uint8)

            # Assign label idx (you can offset by structure type if needed)
            label_volume[data > 0] = idx

    if label_volume is not None:
        out_path = os.path.join(dst_root, f"{patient_id}.nii.gz")
        nib.save(nib.Nifti1Image(label_volume, affine), out_path)
        print(f"✅ Saved: {out_path}")
    else:
        print(f"⚠️ No matching labels found for {patient_id}")

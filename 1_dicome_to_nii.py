import os
import SimpleITK as sitk
from tqdm import tqdm

# ------------------------------------------
# Convert DICOM series to NIfTI
# ------------------------------------------
def convert_dicom_to_nifti(dicom_folder, output_path):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_folder)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    sitk.WriteImage(image, output_path)
    print(f"[DICOM→NIfTI] Saved: {output_path}")

# ------------------------------------------
# Process a single patient case (only DICOM to NIfTI)
# ------------------------------------------
def process_case(patient_id, scan_name, dicom_base, out_image_dir):
    dicom_path = os.path.join(dicom_base, patient_id, scan_name)
    nifti_out = os.path.join(out_image_dir, f"{patient_id}.nii.gz")
    convert_dicom_to_nifti(dicom_path, nifti_out)

# ------------------------------------------
# Traverse patient folders & batch process
# ------------------------------------------
def run_batch(root_dir, output_dir):
    dicom_base = os.path.join(root_dir,"noWatershedSegmentationOrderDICOM")
    out_image_dir = os.path.join(output_dir, "WatershedSegmentationOrderDICOM")
    os.makedirs(out_image_dir, exist_ok=True)

    for patient_id in os.listdir(dicom_base):
        patient_path = os.path.join(dicom_base, patient_id)
        if not os.path.isdir(patient_path):
            continue

        for scan_name in os.listdir(patient_path):
            scan_path = os.path.join(patient_path, scan_name)
            if not os.path.isdir(scan_path):
                continue

            print(f"\n[Processing] {patient_id} → {scan_name}")
            process_case(
                patient_id, scan_name,
                dicom_base,
                out_image_dir
            )

# ------------------------------------------
# MAIN ENTRY POINT
# ------------------------------------------
if __name__ == "__main__":
    root_data_dir = r"E:\linkdoc\Code_work\6Samples"         # 🔁 Update this
    output_dir = r"E:\linkdoc\Code_work\6Samples"       # 🔁 Update this
    run_batch(root_data_dir, output_dir)

import dicom2nifti

def convert_dicom_to_nifti(dicom_dir, output_folder, compression=True, reorient=False):
    dicom2nifti.convert_directory(dicom_dir, output_folder, compression=compression, reorient=reorient)
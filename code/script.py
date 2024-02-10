import os
import argparse
import numpy as np
import pydicom
import SimpleITK as sitk
import dicom2nifti
import glob
import subprocess
import pydicom_seg
import pandas as pd
import nibabel as nib
import warnings

# Deactive Pydicom warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydicom.valuerep")

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def convert_dicom_to_nifti(dicom_dir, output_nifti_path):
    dicom2nifti.convert_directory(dicom_dir, output_nifti_path, compression=True, reorient=False)

def main():
    parser = argparse.ArgumentParser(description='Function that takes as input a Dicom folder of MRI data, segments it with the WMHSynthSeg model and outputs the segmentation in a Dicom folder.')
    parser.add_argument('--dicom_input_dir', required=True, help='Path to input directory with DICOM files (anat)')
    parser.add_argument('--dicom_out_dir', required=True, help='Path to output directory for DICOM files (segmentation)')
    args = parser.parse_args()

    # Create folder
    script_directory = os.path.dirname(os.path.abspath(__file__))
    dicom_input_dir = args.dicom_input_dir
    temp_folder_name = os.path.basename(dicom_input_dir) + "_temp"
    temp_nifti_dir = os.path.join(script_directory, temp_folder_name)
    create_directory(temp_nifti_dir)
    create_directory(args.dicom_out_dir)

    # Convertir DICOM a NIfTI
    convert_dicom_to_nifti(args.dicom_input_dir, temp_nifti_dir)

    # Create Tmp folder and files
    nifti_files = glob.glob(os.path.join(temp_nifti_dir, '*.nii.gz'))
    temp_nifti_mask_1 = os.path.join(temp_nifti_dir, 'WMHSynthSeg_seg.nii.gz')
    temp_nifti_mask_res = os.path.join(temp_nifti_dir, 'res.nii.gz')

    #Processing : (bet2 for brain mask segmentation )
    command_2 = f"python3 WMHSynthSeg/inference.py --i {nifti_files[0]} --o {temp_nifti_mask_1}"
    subprocess.run(command_2, shell=True)

    # Reslincing
    command_3 = f"mri_vol2vol --mov {temp_nifti_mask_1} --targ {nifti_files[0]} --o {temp_nifti_mask_res} --regheader --nearest "
    subprocess.run(command_3, shell=True)

    # Labels names
    data = {'Intensity': [2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26, 28, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 77],
        'Label': [  "Left-Cerebral-White-Matter",
                    "Left-Cerebral-Cortex",
                    "Left-Lateral-Ventricle",
                    "Left-Inf-Lat-Vent",
                    "Left-Cerebellum-White-Matter",
                    "Left-Cerebellum-Cortex",
                    "Left-Thalamus",
                    "Left-Caudate",
                    "Left-Putamen",
                    "Left-Pallidum",
                    "3rd-Ventricle",
                    "4th-Ventricle",
                    "Brain-Stem",
                    "Left-Hippocampus",
                    "Left-Amygdala",
                    "CSF",
                    "Left-Accumbens-area",
                    "Left-VentralDC",
                    "Right-Cerebral-White-Matter",
                    "Right-Cerebral-Cortex",
                    "Right-Lateral-Ventricle",
                    "Right-Inf-Lat-Vent",
                    "Right-Cerebellum-White-Matter",
                    "Right-Cerebellum-Cortex",
                    "Right-Thalamus",
                    "Right-Caudate",
                    "Right-Putamen",
                    "Right-Pallidum",
                    "Right-Hippocampus",
                    "Right-Amygdala",
                    "Right-Accumbens-area",
                    "Right-VentralDC",
                    "WM-hypointensities"]}
    df = pd.DataFrame(data)

    # For dicom metadata
    reader = sitk.ImageSeriesReader()
    dcm_files = reader.GetGDCMSeriesFileNames(f"{args.dicom_input_dir}")
    reader.SetFileNames(dcm_files)
    image = reader.Execute()

    # Open nifti files and re-orient
    image_data_nii = nib.load ( f"{temp_nifti_mask_res}" )
    segmentation_data_array = np.array(image_data_nii.get_fdata())
    segmentation_data_array = np.rot90(segmentation_data_array)
    segmentation_data_array = np.flipud(segmentation_data_array)
    segmentation_data_array = np.transpose(segmentation_data_array, (2, 0, 1))
    all_intensities = {}
    for index, row in df.iterrows():
        intensity = row.iloc[0]
        label_name = row.iloc[1]
        segment = segmentation_data_array == intensity
        all_intensities[intensity] = nib.Nifti1Image(segment.astype(np.uint8), image_data_nii.affine)
        segmentation_data = np.array(all_intensities[intensity].get_fdata())
        template_path = os.path.join(script_directory, 'template', f'{intensity}.json')
        template = pydicom_seg.template.from_dcmqi_metainfo(template_path)
        writer = pydicom_seg.MultiClassWriter( template = template, inplane_cropping=False,  skip_empty_slices=False,  skip_missing_segment=False, )
        if np.max(segmentation_data) != 0:
            segmentation_data_uint = np.asarray(segmentation_data, dtype=np.uint8)
            segmentation = sitk.GetImageFromArray(segmentation_data_uint)
            segmentation.CopyInformation(image)
            source_images = [ pydicom.dcmread(x, stop_before_pixels=True)
                            for x in dcm_files]
            dcm = writer.write(segmentation, source_images)
            output_file_path = os.path.join(args.dicom_out_dir, f"{intensity}_{label_name}_WMH_SynthSeg.dcm")
            dcm.save_as(output_file_path)
        else:
            print(f"Label -  {intensity} -  {label_name} Does Not Exist")
    # For delete temporal (nii) files
    #command_4 = f"rm -r {temp_nifti_dir}"
    #subprocess.run(command_4, shell=True)

if __name__ == "__main__":
    main()

"""

Script to run WMH-SynthSeg model on Dicom inputs and save them as Dicom segmentations.

Authors: Nilser Laines Medina , Nathan Molinier

"""

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

# Deactivate pydicom_seg warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydicom.valuerep")

# Dicom segmentation labels (according to FreeSurferColorLUT.txt)
data = {'Intensity': [1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26, 28, 41, 42, 43, 44, 
                        46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 77],
            'Label': [  "Brain-mask",
                        "Left-Cerebral-White-Matter",
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

# Minimum and maximum number of Dicom files within a folder for the model to apply
min_size = 40
max_size = 380
print('\n')
print(f"The model will only be applied to Dicom folders with a minimum of {min_size} images and a maximum of {max_size} images.")

# ===========================================================================
#                        Functions for the script
# ===========================================================================

def get_last_folders_in_branches(root):
    results = []

    for folder, subfolders, _ in os.walk(root):
        if not subfolders:  # It is a leaf folder
            results.append(folder)
    return results

def count_files_in_folder(folder):
    file_count = sum(1 for file in os.listdir(folder) if os.path.isfile(os.path.join(folder, file)))
    return file_count

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def convert_dicom_to_nifti(dicom_dir, output_nifti_path):
    dicom2nifti.convert_directory(dicom_dir, output_nifti_path, compression=True, reorient=False)

def process_all_subfolders(dicom_input_root, dicom_output_dir):
    last_folders_in_branches = get_last_folders_in_branches(dicom_input_root)
    for last_subfolder in last_folders_in_branches:
        file_count = count_files_in_folder(last_subfolder)
        if min_size <= file_count <= max_size:
            print('')
            print(f" ================ The folder {last_subfolder} have: {file_count} files, The model will be applied ================")
            print('')
            output_folder = os.path.join(dicom_output_dir, last_subfolder)
            input_folder = os.path.join(last_subfolder)
            temp_folder_name = os.path.basename(dicom_output_dir + "_temp")
            temp_folder = os.path.join(temp_folder_name, last_subfolder)
            
            # Creation of temporary folders and output folders
            create_directory(output_folder)
            create_directory(temp_folder)
            script_directory = os.path.dirname(os.path.abspath(__file__))

            # Convertir DICOM a NIfTI
            convert_dicom_to_nifti(input_folder, temp_folder)

            # Temporary folders/files
            nifti_files_all = glob.glob(os.path.join(temp_folder, '*.nii.gz'))
            temp_nifti_mask_1 = os.path.join(temp_folder, '_mask.nii.gz') 
            temp_nifti_mask_res = os.path.join(temp_folder, 'res.nii.gz') 
            
            nifti_files = os.path.join(temp_folder, 'anat.nii.gz') 
            command_1 = f"mv {nifti_files_all[0]}  {nifti_files}"
            subprocess.run(command_1, shell=True)

# ===========================================================================
#                Processing: Segmentation using WMHSynthSeg
# ===========================================================================
            # To test the script, you can try using bet2 to segment only the brain.
            #command_2 = f"python3 docker_file/WMHSynthSeg/inference.py --i {nifti_files} --o {temp_nifti_mask_1}"
            command_2 = f"bet2  {nifti_files}  {temp_folder}/ -m"
            subprocess.run(command_2, shell=True)
            
            # Reslincing of the output (mask) to the anat image
            command_3 = f"mri_vol2vol --mov {temp_nifti_mask_1} --targ {nifti_files} --o {temp_nifti_mask_res} --regheader --nearest "
            subprocess.run(command_3, shell=True)

            # To see input Dicom metadata
            print('')
            reader = sitk.ImageSeriesReader()
            dcm_files = reader.GetGDCMSeriesFileNames(input_folder)
            reader.SetFileNames(dcm_files)
            image = reader.Execute()

            # Re-orient of nifti files 
            image_data_nii = nib.load(f"{temp_nifti_mask_res}")
            segmentation_data_array = np.array(image_data_nii.get_fdata())
            segmentation_data_array = np.rot90(segmentation_data_array) 
            segmentation_data_array = np.flipud(segmentation_data_array) 
            segmentation_data_array = np.transpose(segmentation_data_array, (2, 0, 1))

            # Validation between the number of Dicom images and the anatomical slices.
            num_dcm_files = count_files_in_folder(input_folder)
            if segmentation_data_array.shape[0] == num_dcm_files:
                
                all_intensities = {}
                for index, row in df.iterrows():
                    intensity = row[0]  
                    label_name = row[1]  
                    segment = segmentation_data_array == intensity
                    all_intensities[intensity] = nib.Nifti1Image(segment.astype(np.uint8), image_data_nii.affine)
                    segmentation_data = np.array(all_intensities[intensity].get_fdata()) 
                    template_path = os.path.join(script_directory, 'template', f'{intensity}.json')
                    template = pydicom_seg.template.from_dcmqi_metainfo(template_path)
                    writer = pydicom_seg.MultiClassWriter( template = template, inplane_cropping=False,  skip_empty_slices=False,  skip_missing_segment=False, )
                    
# ===========================================================================
#                        Save segmentation files by class
# ===========================================================================
            
                    if np.max(segmentation_data) != 0:
                        segmentation_data_uint = np.asarray(segmentation_data, dtype=np.uint8)  
                        segmentation = sitk.GetImageFromArray(segmentation_data_uint)
                        segmentation.CopyInformation(image)
                        source_images = [ pydicom.dcmread(x, stop_before_pixels=True)
                                        for x in dcm_files]
                        dcm = writer.write(segmentation, source_images)
                        output_file_path = os.path.join(output_folder, f"{str(intensity).zfill(2)}_{label_name}_WMH_SynthSeg.dcm")
                        dcm.save_as(output_file_path)
                        print('Dicom segmentation sabed on : ',  output_file_path)
                        print('')
                    else:
                        print(f"Label -  {intensity} -  {label_name} Does Not Exist")
                        
                command_4 = f"rm -r {temp_folder_name}"
                subprocess.run(command_4, shell=True)
                print('')
                print(f" Temporary file {temp_folder_name} was deleted" )
                print('')

            else: 
                command_5 = f"rm -r {output_folder}"
                subprocess.run(command_5, shell=True)
                print('')
                print(f"The dicom file {output_folder} has images in the temporary space, so the model will not be applied (possible GRE, DTI, fMRI). ")
                print('')
        else: 
            print('')
            print(f"The {last_subfolder} dicom folder is out of range = {file_count} files")
            print('')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    	description = 
    	'::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\n'
    	'\n'
	    'Function to segment Dicom inputs and save them as Dicom segmentations. \n'
	    'Using: WMH-SynthSeg model, implemented on freesurfer-linux-ubuntu22_x86_64-dev-20240112 \n'
        '\n'
    	'::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\n',
        formatter_class=argparse.RawTextHelpFormatter )
    
    parser.add_argument('--dicom_input_dir', required=True, help='Path to input directory with DICOM files (anat)')
    parser.add_argument('--dicom_output_dir', required=True, help='Path to output directory for DICOM files (segmentation)')
    args = parser.parse_args()
    process_all_subfolders(args.dicom_input_dir, args.dicom_output_dir)
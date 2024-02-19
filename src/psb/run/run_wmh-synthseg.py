# This script is used to perform the inference of the model WMH-SynthSeg (https://surfer.nmr.mgh.harvard.edu/fswiki/WMH-SynthSeg) from FreeSurfer on a given DICOM folder. 
# This same script will return as an output the corresponding DICOM_SEGMENTATION in the output folder.
#
# For more help, please run:  python src/psb/run/run_wmh-synthseg.py -h
#
# Example:
#       python src/psb/run/run_wmh-synthseg.py
#               --dcm-in  ~/<your_dataset>/dicom_anat
#               --dcm-out ~/<your_dataset>/dicom_seg
#
# Authors: Nilser Laines Medina, Nathan Molinier, Julien Cohen-Adad
#
import os
import argparse
import warnings
import subprocess
import glob
import json
import numpy as np
import logging
import coloredlogs

from psb.utils.utils import get_last_folders_in_branches, count_files_in_folder, create_directory, tmp_create, rmtree
from psb.niiXdcm.dcm2nii import convert_dicom_to_nifti
from psb.niiXdcm.nii2dcm import convert_nifti_seg_to_dicom_seg
from psb.utils.image import Image, zeros_like


def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Run wmh_synthseg inference on a dicom folder')
    parser.add_argument('--dcm-in', type=str, required=True, help='Path to input directory with DICOM files (anat)')
    parser.add_argument('--dcm-out', type=str, required=True, help='Path to output directory for DICOM segmentation(s)')
    parser.add_argument('--min-dcm', type=int, default=40, help='Minimum number (int) of slices computed by the model. Default=40')
    return parser


def run_wmh_synthseg():
    parser = get_parser()
    args = parser.parse_args()

    # Set logging level
    logging.basicConfig(level=logging.WARNING)
    coloredlogs.install(fmt='%(message)s', level='WARNING')

    # Load wmh label dictionary
    label_wmh_path = 'src/psb/labels/WMH-SynthSeg/label-maps.json'
    template_dir = 'src/psb/labels/WMH-SynthSeg/template'
    with open(label_wmh_path, "r") as f:
        label_dict = json.load(f)

    # Deactivate pydicom_seg warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="pydicom.valuerep")

    dcm_in = os.path.abspath(args.dcm_in)
    dcm_out = os.path.abspath(args.dcm_out)
    min_dcm = args.min_dcm

    last_folders_in_branches = get_last_folders_in_branches(dcm_in)
    for last_subfolder in last_folders_in_branches:
        file_count = count_files_in_folder(last_subfolder)
        if min_dcm <= file_count :
            print('')
            print(f'================ The folder {last_subfolder} has: {file_count} .dcm files. ================')
            print('')

            # Init paths
            folder_basename = os.path.basename(dcm_in)
            input_folder = os.path.normpath(last_subfolder)

            # Fetch folder structure
            if os.path.basename(input_folder) != folder_basename:
                folder_structure = os.path.join(folder_basename, input_folder.split(f'/{folder_basename}/')[-1])
            else:
                folder_structure = folder_basename
            
            # Create output path
            output_folder = os.path.join(dcm_out, folder_structure)

            # Create output folder if does not exists
            create_directory(output_folder)
            print('output_folder !:', output_folder)

            # Create a temporary folder
            temp_folder_name = os.path.basename(os.path.normpath(dcm_out) + "_temp")
            tmpdir = tmp_create(basename=temp_folder_name)

            # Convert DICOM to NIfTI
            convert_dicom_to_nifti(input_folder, tmpdir, reorient=False)
            nifti_files_all = glob.glob(os.path.join(tmpdir, '*.nii.gz'))

            if len(nifti_files_all) > 1:
                logging.warning('Multiple images were detected')
            
            # For each generated files
            for nifti_anat_path in nifti_files_all:

                # Create temporary files paths
                temp_dseg = os.path.join(tmpdir, 'dseg.nii.gz') 
                temp_dseg_res = os.path.join(tmpdir, 'dseg_res.nii.gz') 

                print('')
                print(f'=========================== Starting inference with WMH-SynthSeg ==========================')
                print('')

                # To test the script, you can try using bet2 to segment only the brain.
                command_1 = f"python3 /usr/local/WMHSynthSeg/inference.py --i {nifti_anat_path} --o {temp_dseg}"
                
                # Run inference using a subprocess
                subprocess.run(command_1, shell=True)

                # Reslincing of the output (mask) to the anat image
                command_2 = f"mri_vol2vol --mov {temp_dseg} --targ {nifti_anat_path} --o {temp_dseg_res} --regheader --nearest "
                # Run inference using a subprocess
                subprocess.run(command_2, shell=True)

                # Load image
                image_out_nii = Image(temp_dseg_res)

                # Validation between the number of Dicom images and the anatomical slices.
                num_dcm_files = count_files_in_folder(input_folder)
                if image_out_nii.data.shape[2] == num_dcm_files:
                    # Split the multiple discrete segmentation (dseg)
                    for key, val in label_dict.items():
                        intensity = val
                        label_name = key
                        mask = zeros_like(image_out_nii)
                        mask.data = (image_out_nii.data).astype(np.uint8)   
                        mask.data[np.where(image_out_nii.data != intensity)] = 0
                        template_path = os.path.join(template_dir, f'{label_name}.json')
                        # Save each class in different files
                        if np.max(mask.data) != 0:
                            output_file_path = os.path.join(output_folder, f"{str(intensity).zfill(2)}_{label_name}_WMH_SynthSeg.dcm")
                            dcm_seg_file = convert_nifti_seg_to_dicom_seg(input_folder, mask, template_path)
                            dcm_seg_file.save_as(output_file_path)
                            print('DICOM segmentation saved on : ',  output_file_path)
                            print('')
                        else:
                            print(f"Label - {intensity} - {label_name} Does Not Exist")
                        
                    # Delete the temporary folder
                    rmtree(tmpdir)

                    print('')
                    print(f" Temporary file {temp_folder_name} was deleted" )
                    print('')

                else: 
                    logging.warning(f'Different number of slices with the original DICOM (possible GRE, DTI, fMRI).')
        else:
            if min_dcm > file_count:
                logging.warning(f"The dicom folder must contain at least {min_dcm} files: {file_count} files were detected. If you wish to run the script with fewer files, please use the flag --min-dcm")


if __name__ == "__main__":
    run_wmh_synthseg()
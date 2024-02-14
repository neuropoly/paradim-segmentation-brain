import os
import argparse
import warnings
import subprocess
import glob
import json
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import pydicom_seg
import pydicom
from nibabel.nicom.dicomwrappers import wrapper_from_file

from psb.utils.utils import get_last_folders_in_branches, count_files_in_folder, create_directory, tmp_create, rmtree
from psb.utils.image import Image
from psb.niiXdcm.dcm2nii import convert_dicom_to_nifti


def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Run wmh_synthseg inference on a dicom folder')
    parser.add_argument('--dcm-in', type=str, required=True, help='Path to input directory with DICOM files (anat)')
    parser.add_argument('--dcm-out', type=str, required=True, help='Path to output directory for DICOM segmentation(s)')
    parser.add_argument('--min-dcm', type=int, default=40, help='Minimum number (int) of slices computed by the model. Default=40')
    parser.add_argument('--max-dcm', type=int, default=380, help='Maximum number (int) of slices computed by the model. Default=380')
    return parser


def run_wmh_synthseg():
    parser = get_parser()
    args = parser.parse_args()

    # Load wmh label dictionary
    label_wmh_path = 'src/psb/labels/WMH-SynthSeg/label-dict.json'
    template_dir = 'src/psb/labels/WMH-SynthSeg/template'
    with open(label_wmh_path, "r") as f:
        label_dict = json.load(f)

    # Deactivate pydicom_seg warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="pydicom.valuerep")

    dcm_in = os.path.abspath(args.dcm_in)
    dcm_out = os.path.abspath(args.dcm_out)
    min_dcm = args.min_dcm
    max_dcm = args.max_dcm

    last_folders_in_branches = get_last_folders_in_branches(dcm_in)
    for last_subfolder in last_folders_in_branches:
        file_count = count_files_in_folder(last_subfolder)
        if min_dcm <= file_count <= max_dcm:
            print('')
            print(f'================ The folder {last_subfolder} has: {file_count} .dcm files. ================')
            print('')

            # Init paths
            folder_basename = os.path.basename(dcm_in)
            folder_structure = os.path.join(folder_basename,last_subfolder.split(f'/{folder_basename}/')[-1])
            input_folder = os.path.normpath(last_subfolder)
            output_folder = os.path.join(dcm_out, folder_structure)
            script_directory = os.path.dirname(os.path.abspath(__file__))

            # Create output folder if does not exists
            create_directory(output_folder)
            
            # Create a temporary folder
            temp_folder_name = os.path.basename(os.path.normpath(dcm_out) + "_temp")
            tmpdir = tmp_create(basename=temp_folder_name)

            # Convert DICOM to NIfTI
            convert_dicom_to_nifti(input_folder, tmpdir)
            nifti_files_all = glob.glob(os.path.join(tmpdir, '*.nii.gz'))

            if len(nifti_files_all) > 1:
                print('Multiple images were detected')
            
            # For each generated files
            for nifti_anat_path in nifti_files_all:

                # Create temporary files paths
                temp_dseg = os.path.join(tmpdir, 'dseg.nii.gz') 
                temp_dseg_res = os.path.join(tmpdir, 'res.nii.gz') 

                print('')
                print(f'=========================== Starting inference with WMH-SynthSeg ==========================')
                print('')

                # To test the script, you can try using bet2 to segment only the brain.
                # command_1 = f"python3 docker_file/WMHSynthSeg/inference.py --i {nifti_anat_path} --o {temp_dseg}"
                #command_1 = 'ls'

                # Run inference using a subprocess
                #subprocess.run(command_1, shell=True)

                # Reslincing of the output (mask) to the anat image
                #command_2 = f"mri_vol2vol --mov {temp_dseg} --targ {nifti_anat_path} --o {temp_dseg_res} --regheader --nearest "
                #subprocess.run(command_2, shell=True)

                # To see input Dicom metadata
                print('')
                reader = sitk.ImageSeriesReader()
                dcm_files = reader.GetGDCMSeriesFileNames(input_folder)
                reader.SetFileNames(dcm_files)
                image = reader.Execute()

                # Re-orient of nifti files 
                # TODO: make this conversion more reliable:
                # - extract the orientation from the original DICOM
                # - use the class Image
                image_data_nii = nib.load(temp_dseg_res)
                segmentation_data_array = np.array(image_data_nii.get_fdata())
                segmentation_data_array = np.rot90(segmentation_data_array) 
                segmentation_data_array = np.flipud(segmentation_data_array) 
                segmentation_data_array = np.transpose(segmentation_data_array, (2, 0, 1))

                # Validation between the number of Dicom images and the anatomical slices.
                num_dcm_files = count_files_in_folder(input_folder)
                if segmentation_data_array.shape[0] == num_dcm_files:
                    all_intensities = {}
                    # Split the multiple discrete segmentation (dseg)
                    for key, val in label_dict.items():
                        intensity = val
                        label_name = key 
                        segment = segmentation_data_array == intensity
                        all_intensities[intensity] = nib.Nifti1Image(segment.astype(np.uint8), image_data_nii.affine)
                        segmentation_data = np.array(all_intensities[intensity].get_fdata()) 
                        template_path = os.path.join(template_dir, f'{label_name}.json')
                        template = pydicom_seg.template.from_dcmqi_metainfo(template_path)
                        writer = pydicom_seg.MultiClassWriter( template = template, inplane_cropping=False,  skip_empty_slices=False,  skip_missing_segment=False, )
                    
                        # Save each class in different files
                        if np.max(segmentation_data) != 0:
                            segmentation_data_uint = np.asarray(segmentation_data, dtype=np.uint8)  
                            segmentation = sitk.GetImageFromArray(segmentation_data_uint)
                            segmentation.CopyInformation(image)
                            source_images = [pydicom.dcmread(x, stop_before_pixels=True)
                                            for x in dcm_files]
                            dcm = writer.write(segmentation, source_images)
                            output_file_path = os.path.join(output_folder, f"{str(intensity).zfill(2)}_{label_name}_WMH_SynthSeg.dcm")
                            dcm.save_as(output_file_path)
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
                    raise ValueError(f'Different number of slices with the original DICOM (possible GRE, DTI, fMRI).')
        else:
            if min_dcm > file_count:
                raise ValueError(f"The dicom folder must contain at least {min_dcm} files: {file_count} files were detected")
            else:
                raise ValueError(f"The dicom folder must contain less than {max_dcm} files: {file_count} files were detected")


if __name__ == "__main__":
    run_wmh_synthseg()
import os
import argparse
import numpy as np
import pydicom
import SimpleITK as sitk
import dicom2nifti
import glob
import subprocess

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def read_dicom_metadata(ruta):
    ds = pydicom.dcmread(ruta, force=True)
    return ds

def convert_dicom_to_nifti(dicom_dir, output_nifti_path):
    dicom2nifti.convert_directory(dicom_dir, output_nifti_path, compression=True, reorient=False)

def write_slices(series_tag_values, new_img, i, out_dir,dicom_dir):
    image_slice = new_img[:, :, i]
    image_slice = sitk.Cast(image_slice, sitk.sitkInt16)
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    
    list(map(lambda tag_value: image_slice.SetMetaData(tag_value[0], tag_value[1]), series_tag_values))
    
    dicom_folder = os.listdir(dicom_dir)
    dcm_files = [dcm_files for dcm_files in dicom_folder if dcm_files.lower().endswith(".dcm")]
    for dcm_files in dicom_folder:
        dcm_files
    dicom_folder[0]
    root_one_dcm = dicom_dir + dicom_folder[0]   
    
    dicom_metadata = read_dicom_metadata(root_one_dcm)
    #value_0008_0012 = dicom_metadata.get((0x0008, 0x0012), "Unknown").value
    value_0008_0020 = dicom_metadata.get((0x0008, 0x0020), "Unknown").value
    value_0008_0060 = dicom_metadata.get((0x0008, 0x0060), "Unknown").value
    #value_0020_0032 = dicom_metadata.get((0x0020, 0x0032), "Unknown").value
    value_0020_0013 = dicom_metadata.get((0x0020, 0x0013) ).value

    #image_slice.SetMetaData("0008|0012", value_0008_0012)  
    image_slice.SetMetaData("0008|0020", value_0008_0020)  
    image_slice.SetMetaData("0008|0060", value_0008_0060) 
    image_slice.SetMetaData("0020|0032", '\\'.join(map(str, new_img.TransformIndexToPhysicalPoint((0, 0, i)))))  # Image Position (Patient)
    image_slice.SetMetaData("0020,0013", str(value_0020_0013))  # Instance Number
    
    writer.SetFileName(os.path.join(out_dir, f'slice{i:04d}.dcm'))
    writer.Execute(image_slice) 

def convert_nifti_to_dicom(in_dir, out_dir, dicom_dir):
    new_img = sitk.ReadImage(in_dir)

    dicom_folder = os.listdir(dicom_dir)
    dcm_files = [dcm_files for dcm_files in dicom_folder if dcm_files.lower().endswith(".dcm")]
    for dcm_files in dicom_folder:
        dcm_files
    dicom_folder[0]
    root_one_dcm = dicom_dir + dicom_folder[0]   

    dicom_metadata = read_dicom_metadata(root_one_dcm)
    value_0008_0031 = dicom_metadata.get((0x0008, 0x0031), "Unknown").value
    value_0008_0021 = dicom_metadata.get((0x0008, 0x0021), "Unknown").value
    value_0008_0008 = dicom_metadata.get((0x0008, 0x0008), "Unknown").value
    value_0020_000e = dicom_metadata.get((0x0020, 0x000e), "Unknown").value
    value_0020_103e = dicom_metadata.get((0x0008, 0x103e), "Unknown").value
    
    direction = new_img.GetDirection()   

    series_tag_values = [("0008|0031", value_0008_0031),  # Series Time
                         ("0008|0021", value_0008_0021),  # Series Date
                         ("0008|0008", '\\'.join(value_0008_0008)),  # Image Type
                         ("0020|000e", value_0020_000e),  # Series Instance UID
                         ("0020|0037", '\\'.join(map(str, (direction[0], direction[3], direction[6],
                                                           direction[1], direction[4], direction[7])))),

                         ("0008|103e", value_0020_103e)]  # Series Description
    
    for i in range(new_img.GetDepth()):
        write_slices(series_tag_values, new_img, i, out_dir,dicom_dir)

def main():
    parser = argparse.ArgumentParser(description='Converts DICOM to NIfTI and vice versa.')
    parser.add_argument('--dicom_dir', required=True, help='Path to directory with DICOM files')
    parser.add_argument('--dicom_out_dir', required=True, help='Path to output directory for DICOM files')
    
    args = parser.parse_args()
    # Create folder        
    temp_nifti_dir = "nifti_files/"
    create_directory(temp_nifti_dir)
    create_directory(args.dicom_out_dir)
    
    # Convertir DICOM a NIfTI
    convert_dicom_to_nifti(args.dicom_dir, temp_nifti_dir)
    
    temp_nifti_file = "nifti_files/*.nii.gz"
    nifti_files = glob.glob(os.path.join(temp_nifti_file))   
    #command_1 = f"bet2  {nifti_files[0]}  {temp_nifti_dir} -m"
    temp_nifti_mask_1 = "nifti_files/_mask.nii.gz"
    command_1 = f"mri_convert -vs 2 2 2 {nifti_files[0]}  {temp_nifti_mask_1}"
    #command_1 = f"python3 docker_file/WMHSynthSeg/inference.py --i {nifti_files[0]} --o {temp_nifti_mask_1}"
    subprocess.run(command_1, shell=True)
    # Convertir NIfTI a DICOM
    temp_nifti_mask = "nifti_files/*_mask.nii.gz"
    nifti_files_mask = glob.glob(os.path.join(temp_nifti_mask)) 
    convert_nifti_to_dicom(nifti_files_mask[0], args.dicom_out_dir, args.dicom_dir)

if __name__ == "__main__":
    main()
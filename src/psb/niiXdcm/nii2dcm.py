import SimpleITK as sitk
import pydicom
import pydicom_seg
import numpy as np

def convert_nifti_seg_to_dicom_seg(dcm_path_input , seg_nii_array , pydicom_seg_writer):
    reader = sitk.ImageSeriesReader()
    dcm_files = reader.GetGDCMSeriesFileNames(dcm_path_input)
    reader.SetFileNames(dcm_files)
    dcm_reader = reader.Execute()
    seg_array_uint = np.asarray(seg_nii_array, dtype=np.uint8)  
    seg_sitk = sitk.GetImageFromArray(seg_array_uint)
    seg_sitk.CopyInformation(dcm_reader)
    dcm_source_images = [pydicom.dcmread(x, stop_before_pixels=True)
                    for x in dcm_files]
    return pydicom_seg_writer.write(seg_sitk, dcm_source_images)
    
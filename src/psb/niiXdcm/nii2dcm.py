import SimpleITK as sitk
import pydicom
import pydicom_seg
import numpy as np


def convert_nifti_seg_to_dicom_seg(dcm_path_input, seg_image, template_path):
    '''
    :param dcm_path_input: DICOM input folder used to extract metadata
    :param seg_image: Image object
    :param template_path: template path
    '''
    template = pydicom_seg.template.from_dcmqi_metainfo(template_path)
    writer = pydicom_seg.MultiClassWriter(template=template, inplane_cropping=False, skip_empty_slices=False, skip_missing_segment=False)
    reader = sitk.ImageSeriesReader()
    dcm_files = reader.GetGDCMSeriesFileNames(dcm_path_input)
    reader.SetFileNames(dcm_files)
    dcm_reader = reader.Execute()
    
    # Change orientation itksnap
    seg_image.change_orientation(reverse_orientation_itksnap(seg_image.orientation))

    # Create dicom_seg object
    seg_sitk = sitk.GetImageFromArray(seg_image.data)
    seg_sitk.CopyInformation(dcm_reader)

    dcm_source_images = [pydicom.dcmread(x, stop_before_pixels=True) for x in dcm_files]
    return writer.write(seg_sitk, dcm_source_images)


def reverse_orientation_itksnap(orientation):
    return orientation[::-1]


def create_nifti_orientation_matrix(orientation):
    """
    From an orientation (e.g. ASL) create the nifti rotation matrix
    """
    nifti_ori_matrix = np.zeros((3, 3))
    for idx, letter in enumerate(orientation):
        if letter == 'L':
            nifti_ori_matrix[0, idx] = 1
        elif letter == 'R':
            nifti_ori_matrix[0, idx] = -1
        elif letter == 'P':
            nifti_ori_matrix[1, idx] = 1
        elif letter == 'A':
            nifti_ori_matrix[1, idx] = -1
        elif letter == 'I':
            nifti_ori_matrix[2, idx] = 1
        elif letter == 'S':
            nifti_ori_matrix[2, idx] = -1
    return nifti_ori_matrix

    
import os
import dicom2nifti
import pydicom
import numpy as np
import glob

from psb.utils.image import Image

def convert_dicom_to_nifti(dicom_dir, output_folder, compression=True, reorient=False):
    dicom2nifti.convert_directory(dicom_dir, output_folder, compression=compression, reorient=reorient)
    dcm_ori_matrix = get_orientation_matrix_from_dicom(dicom_dir)
    nifti_ori_matrix = get_nifti_orientation_matrix(dcm_ori_matrix)
    orig_orientation = read_orientation(nifti_ori_matrix)
    nifti_files = glob.glob(os.path.join(output_folder, '*.nii.gz'))
    for file_path in nifti_files:
        img = Image(file_path)
        nifti_orientation = img.orientation
        if nifti_orientation != orig_orientation:
            img.change_orientation(orig_orientation)
            img.save(file_path)


def read_orientation(ori_matrix, convention='nifti'):
    """
    Read orientation from an orientation matrix
    """
    orientation = ''
    if convention == 'nifti':
        for ax in np.hsplit(ori_matrix,3):
            if ax[0] == 1:
                orientation += 'L'
            elif ax[0] == -1:
                orientation += 'R'
            elif ax[1] == 1:
                orientation += 'P'
            elif ax[1] == -1:
                orientation += 'A'
            elif ax[2] == 1:
                orientation += 'I'
            elif ax[2] == -1:
                orientation += 'S'
            else:
                raise ValueError(f'Error with matrix: {ori_matrix}')
    
    elif convention == 'dicom':
        for ax in np.hsplit(ori_matrix,3):
            if ax[0] == 1:
                orientation += 'R'
            elif ax[0] == -1:
                orientation += 'L'
            elif ax[1] == 1:
                orientation += 'A'
            elif ax[1] == -1:
                orientation += 'P'
            elif ax[2] == 1:
                orientation += 'I'
            elif ax[2] == -1:
                orientation += 'S'
            else:
                raise ValueError(f'Error with matrix: {ori_matrix}')
            
    else:
        raise ValueError(f'Unknown convention: {convention}')

    return orientation


def read_dicom_metadata(dicom_dir):
    """
    From a DICOM folder, read the metadata of the first .dcm. The metadata should be the same accross files
    """
    dcm_file = os.path.join(dicom_dir,os.listdir(dicom_dir)[0]) # Get metadata from 1 dicom
    ds = pydicom.dcmread(dcm_file, force=True)
    return ds


def get_orientation_matrix_from_dicom(dicom_dir):
    """
    From an input DICOMN file, extract the orientation matrix and return the rounded matrix
    """
    dicom_metadata = read_dicom_metadata(dicom_dir)
    Ax, Ay, Az, Bx, By, Bz = dicom_metadata.get((0x0020,0x0037)).value
    dcm_ori_matrix = np.array([[Ax, Ay, Az],[Bx, By, Bz]]).transpose()
    C = np.expand_dims(np.cross(dcm_ori_matrix[:,0],dcm_ori_matrix[:,1]), axis=-1)
    dcm_ori_matrix = np.around(np.concatenate((dcm_ori_matrix, C), axis=1))
    return dcm_ori_matrix


def get_nifti_orientation_matrix(dcm_ori_matrix):
    """
    NIfTI coordinate system:
    X: L --> R
    Y: P --> A
    Z: I --> S
    DICOM coordinate system:
    X: R --> L
    Y: A --> P
    Z: I --> S
    """
    v = np.expand_dims(np.array([-1, -1, 1]), axis=1)
    nifti_ori_matrix = dcm_ori_matrix * v
    return nifti_ori_matrix

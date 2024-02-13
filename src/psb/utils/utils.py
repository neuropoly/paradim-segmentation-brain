import os
import logging
import datetime
import tempfile
import datetime
import shutil

from psb.niiXdcm.dcm2nii import convert_dicom_to_nifti

logger = logging.getLogger(__name__)

def get_last_folders_in_branches(root):
    results = []

    for folder, subfolders, _ in os.walk(root):
        if not subfolders:  # It is a leaf folder
            results.append(folder)
    return results


def count_files_in_folder(folder):
    return len([file for file in os.listdir(folder) if file.endswith('.dcm')])


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def tmp_create(basename):
    """Create temporary folder and return its path

    Copied from https://github.com/spinalcordtoolbox/spinalcordtoolbox/
    """
    prefix = f"{basename}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    tmpdir = tempfile.mkdtemp(prefix=prefix)
    logger.info(f"Creating temporary folder ({tmpdir})")
    return tmpdir


def rmtree(folder):
    """Recursively remove folder

    Copied from https://github.com/spinalcordtoolbox/spinalcordtoolbox/
    """
    shutil.rmtree(folder)


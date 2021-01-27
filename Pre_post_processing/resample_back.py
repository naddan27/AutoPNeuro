import os
import errno
import shutil
import numpy as np
import nibabel as nib

from subprocess import call

import os
import multiprocessing

from joblib import Parallel, delayed
from preprocessing_functions import *

#path to nifti directory
nifti_dir = '/home/neural_network_code/Data/Patients/'
vols_to_process = ['t1ce_RAI_RESAMPLED_REG_N4_SS_NORM.nii.gz']
rois_to_process = ['t1ce_prediction.nii.gz']
original = 't1ce.nii.gz'

#path to SLICER3D (used for reorienting, registering, resampling volumes)
slicer_dir = '/home/shared_software/Slicer-4.10.2-linux-amd64/Slicer'
#path to ROBEX (used for skull stripping volumes)
robex_dir = '/home/shared_software/ROBEX/runROBEX.sh'

# parameter to run orientation module
orientation = 'RAI'
# parameters to run resampling module
interp_type_vol = 'bspline'
interp_type_roi = 'nearestNeighbor'
# parameters to run registration module (will register T2 to T1 image)
transform_mode = 'Off'
transform_type='Rigid,ScaleVersor3D,ScaleSkewVersor3D,Affine'
interpolation_mode = 'BSpline'
sampling_percentage = .02
affine_transform_filename = 'affine_transform.h5'
# parameter to run bias correction module
n4_iterations = [50,50,30,20]
# parameter to run skull strip module (use T1 volume for skull stripping)
volume_to_skullstrip = vols_to_process[0][:-7] + '_'

#####################################################################################
#run preprocessing over all patients
patients = nested_folder_filepaths(nifti_dir, vols_to_process)
patients.sort()

def put_back_into_original_dimension(patient):
    #get the spacing of the original dimension
    original_img = nib.load(os.path.join(nifti_dir, patient, original))
    spacing = ",".join([str(original_img.header.get_zooms()[i]) for i in range(3)])

    # 1) reorient volumes and rois
    reoriented_volumes = reorient_volume(nifti_dir, patient, vols_to_process, orientation, slicer_dir)
    if len(rois_to_process) > 0:
        #first figure out if ground truth is on T2 or FLAIR
        reoriented_rois = reorient_volume(nifti_dir, patient, rois_to_process, orientation, slicer_dir)
    # 2) resample to isotropic resolution
    resampled_volumes = resample_volume(nifti_dir, patient, reoriented_volumes, spacing, interp_type_vol, slicer_dir)
    if len(rois_to_process) > 0:
        #resample ROI using reference modality
        resampled_rois = resample_volume_using_reference(nifti_dir, patient, reoriented_rois, resampled_volumes[0], interp_type_roi, slicer_dir)
    # 3) register all patients to reference modality (i.e. modality on which ground truth was performed)
    registered_volumes = register_volume(nifti_dir, patient, resampled_volumes[0], resampled_volumes, transform_mode, transform_type, interpolation_mode, sampling_percentage, affine_transform_filename, slicer_dir)
    if len(rois_to_process) > 0:
        # use found affine to register other label maps to input space
        resampled_rois_reference = resampled_rois.pop(0)
        registered_rois = resample_volume_using_reference(nifti_dir, patient, resampled_rois, None, interp_type_roi, slicer_dir, output_transform_filename=affine_transform_filename, append_tag='')
        registered_rois.insert(0, resampled_rois_reference)
    # 4) correct size differences between volumes here to prevent downstream issues
    ground_truth_reference_volume = registered_volumes.pop(0)
    registered_volumes = resample_volume_using_reference(nifti_dir, patient, registered_volumes, ground_truth_reference_volume, interp_type_vol, slicer_dir, append_tag='')
    registered_volumes.insert(0, ground_truth_reference_volume)


num_cores = multiprocessing.cpu_count()
Parallel(n_jobs=num_cores)(delayed(put_back_into_original_dimension)(patient) for patient in patients)
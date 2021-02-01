import multiprocessing
import warnings
import nibabel as nib
import numpy as np
import contextlib
import joblib
import pandas as pd
from joblib import Parallel, delayed
import os
import random
from functools import partial
from calculate_RAPNO import get_rapno 
from random import sample, seed
from tqdm import tqdm
from math import floor
import itertools


rapno_scores_q = multiprocessing.Queue()
rapno_file_paths_q = multiprocessing.Queue()

"""
Return list of rapno scores that correlate with each file path in inputs

Inputs:
- all_file_paths - list of Strings, file paths containing the segmentation
- all_keys - list of int, key value that represents positive class
"""
def get_rapno_multiple_segmentation(total_num_to_process, num_lesions, all_keys):
    process_completed = False 

    while not process_completed:
        id_file_path_list = rapno_file_paths_q.get()

        if id_file_path_list == "END":
            process_completed = True
            break
        else:
            assert len(id_file_path_list[1]) == len(all_keys), "Length of keys does not match number of file paths"
            scores = []

            for fp, key in zip(id_file_path_list[1], all_keys):
                tumor_img = nib.load(fp)
                tumormask = tumor_img.get_data()
                header = tumor_img.header
                vox_x = header.get_zooms()[0]
                vox_z = header.get_zooms()[2]

                binary_mask = (tumormask == key) * 1

                rano_measurement = get_rapno(binary_mask,vox_x,vox_z,None,None,num_lesions, False)
                scores.append(rano_measurement)
            scores.insert(0, id_file_path_list[0])
            
            rapno_scores_q.put(scores)

            print("/".join(id_file_path_list[1][0].split("/")[-4:-2]), total_num_to_process, os.getpid())

def get_rapno_multiple_segmentation_all_lesions(total_num_to_process, num_lesions, all_keys):
    process_completed = False 

    while not process_completed:
        id_file_path_list = rapno_file_paths_q.get()

        if id_file_path_list == "END":
            process_completed = True
            break
        else:
            assert len(id_file_path_list[1]) == len(all_keys), "Length of keys does not match number of file paths"
            scores = []

            for fp, key in zip(id_file_path_list[1], all_keys):
                tumor_img = nib.load(fp)
                tumormask = tumor_img.get_data()
                header = tumor_img.header
                vox_x = header.get_zooms()[0]
                vox_z = header.get_zooms()[2]

                if len(tumormask.shape) == 4:
                    tumormask = tumormask[:,:,:,0]
                elif len(tumormask.shape) > 4:
                    raise AssertionError("check shape of " + str(fp))
                tumormask = tumormask.round()

                binary_mask = (tumormask == key) * 1

                rano_measurement = get_rapno(binary_mask,vox_x,vox_z,None,None,num_lesions, False, all_lesions = True)
                for x in rano_measurement:
                    scores.append(x)
            scores.insert(0, id_file_path_list[0])
            
            rapno_scores_q.put(scores)

            print("/".join(id_file_path_list[1][0].split("/")[-4:-2]), total_num_to_process, os.getpid())

"""
Return 2D array where each row has the first element as the
patient ID and the second element as a list of file paths that correspond
to the labels specified matching the patient. If the patient does not have
the labels of interest, skip patient

Inputs:
- base_dir: String, file path of the base directory
  - base_dir has to be the direct folder above folder containing segmentations
  - if file structure is patientID>output>label.nii, make label 'output/label.nii'
- all_labels: list, element represents the name of the mask
  - e.g. ['T1S.nii.gz', 'segmentation.nii.gz']
- random_sample: None, float, represents the proportion to randomly sample
  - if None, calculate for all patients
"""
def get_file_paths_of_labels(base_dir, all_labels, random_sample = None, random_seed = None):
    all_patient_ids = os.listdir(base_dir)

    all_file_paths = [[os.path.join(base_dir, patient_id, label) for label in all_labels] for patient_id in all_patient_ids]

    id_file_paths = []
    for id, fps in zip(all_patient_ids, all_file_paths):
        files_exist = True
        for fp in fps:
            #check if the file exists. If not, skips this patient

            if not os.path.exists(fp):
                print(fp, " does not exist. Skipping this patient")
                files_exist = False
        
        if files_exist:
            id_file_paths.append([id, fps])

    if random_sample == None:
        return np.array(id_file_paths)
    
    #get a random sample 
    seed(random_seed)

    if random_sample != None:
        return np.array(sample(id_file_paths, int(random_sample*len(id_file_paths))))
    return np.array(id_file_paths)

"""
Return df with the RAPNO scores of each patient for every label

Inputs:
- base_dir: String, file path of the base directory
- all_labels: list, element represents the name of the mask
- all_keys: list of int, key value that represents positive class
- random_sample: None, float, represents the proportion to randomly sample
  - if None, calculate for all patients
- random_seed: Int, represents the seed to create same random sample
"""
def rapno_to_df(base_dir, all_labels, all_keys, num_lesions, random_sample = None, random_seed = None):
    #make sure the rapno queue is empty before starting
    while not rapno_scores_q.empty():
        rapno_scores_q.get()
    while not rapno_file_paths_q.empty():
        rapno_file_paths_q.get()

    #get all of the file paths of the images to get rapno
    id_file_paths = get_file_paths_of_labels(base_dir, all_labels, random_sample=random_sample, random_seed = random_seed)
    id_file_path_list = id_file_paths.tolist()
    random.shuffle(id_file_path_list)

    #put the file paths into the queue
    num_processes = multiprocessing.cpu_count() - 1
    for id_file_path in id_file_path_list:
        rapno_file_paths_q.put(id_file_path)
    for i in range(num_processes):
        rapno_file_paths_q.put("END")

    parallel_get_rapno = partial(get_rapno_multiple_segmentation, all_keys = all_keys)

    processes = []
    for i in range(num_processes):
        num = len(id_file_paths)
        t = multiprocessing.Process(target=parallel_get_rapno, args=(num, num_lesions))
        processes.append(t)
        t.start()
    
    for one_process in processes:
        one_process.join()

    #check if the file path queue is empty. If not, print it out
    while not rapno_file_paths_q.empty():
        print(rapno_file_paths_q.get() )

    rapno_as_list = []
    while not rapno_scores_q.empty():
        rapno_as_list.append(rapno_scores_q.get())

    rapno_as_np_array = np.array(rapno_as_list)

    column_labels = ['Patient_id']
    column_labels.extend(all_labels)
    rapno_df = pd.DataFrame(data=rapno_as_np_array, columns = column_labels)
    rapno_df.sort_values(by=['Patient_id'])

    return rapno_df

def rapno_to_df_all_lesions(base_dir, all_labels, all_keys, num_lesions, random_sample = None, random_seed = None):
    #make sure the rapno queue is empty before starting
    while not rapno_scores_q.empty():
        rapno_scores_q.get()
    while not rapno_file_paths_q.empty():
        rapno_file_paths_q.get()

    #get all of the file paths of the images to get rapno
    id_file_paths = get_file_paths_of_labels(base_dir, all_labels, random_sample=random_sample, random_seed = random_seed)
    id_file_path_list = id_file_paths.tolist()
    random.shuffle(id_file_path_list)

    #put the file paths into the queue
    num_processes = multiprocessing.cpu_count() - 1
    for id_file_path in id_file_path_list:
        rapno_file_paths_q.put(id_file_path)
    for i in range(num_processes):
        rapno_file_paths_q.put("END")

    parallel_get_rapno = partial(get_rapno_multiple_segmentation_all_lesions, all_keys = all_keys)

    processes = []
    for i in range(num_processes):
        num = len(id_file_paths)
        t = multiprocessing.Process(target=parallel_get_rapno, args=(num, num_lesions))
        processes.append(t)
        t.start()
    
    for one_process in processes:
        one_process.join()

    #check if the file path queue is empty. If not, print it out
    while not rapno_file_paths_q.empty():
        print(rapno_file_paths_q.get() )

    rapno_as_list = []
    while not rapno_scores_q.empty():
        rapno_as_list.append(rapno_scores_q.get())

    rapno_as_np_array = np.array(rapno_as_list)

    column_labels = ['Patient_id']
    
    lesion_labels = []
    for al in all_labels:
        for i in range(num_lesions):
            lesion_labels.append(str(al) + "_" + str(i + 1))
    column_labels.extend(lesion_labels)
    
    rapno_df = pd.DataFrame(data=rapno_as_np_array, columns = column_labels)
    rapno_df.sort_values(by=['Patient_id'])

    return rapno_df
"""
Write csv with the RAPNO scores of each patient for every label

Inputs:
- base_dir: String, file path of the base directory
- export_to: String, file path of the export_to directory
- all_labels: list, element represents the name of the mask
- all_keys: list of int, key value that represents positive class
- random_sample: None, float, represents the proportion to randomly sample
  - if None, calculate for all patients
- random_seed: Int, represents the seed to create same random sample

Returns:
path of the csv file
"""
def rapno_to_csv_all_lesions(base_dir, export_to, all_labels, all_keys, num_lesions, random_sample = None, random_seed = None):
    os.makedirs(export_to, exist_ok = True) #make directory if it does not exist 
    rapno_df = rapno_to_df_all_lesions(base_dir, all_labels, all_keys, num_lesions, random_sample = random_sample, random_seed = random_seed)
    csv_fp = os.path.join(export_to,  "rapno_scores_all_lesions.csv")
    rapno_df.to_csv(csv_fp)
    return csv_fp

def rapno_to_csv(base_dir, export_to, all_labels, all_keys, num_lesions, random_sample = None, random_seed = None):
    os.makedirs(export_to, exist_ok = True) #make directory if it does not exist 
    rapno_df = rapno_to_df(base_dir, all_labels, all_keys, num_lesions, random_sample = random_sample, random_seed = random_seed)
    csv_fp = os.path.join(export_to,  "rapno_scores.csv")
    rapno_df.to_csv(csv_fp)
    return csv_fp

"""
Allows for the use of tqdm with parallel processes
"""
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close() 

"""
Return dice score between two masks

Inputs:
- mask1, np.array that represents 1st segmentation mask we are comparing with dice
- mask2, np.array that represents 2nd segmentaiton mask we are comparing with dice
"""
def calculate_dice_as_np_arrays(mask1, mask2):
    if np.any(mask1) or np.any(mask2):
        mask1_flat = mask1.flatten()
        mask2_flat = mask2.flatten()
        intersection = np.sum(np.multiply(mask1_flat, mask2_flat))
        diceCoef = (2. * intersection) / (np.sum(mask1_flat) + np.sum(mask2_flat))
    else:
        diceCoef = 1
    
    return diceCoef


"""
Return the dice score between 2 segmentations of a patient

Inputs:
- all_file_paths - list of Strings, file paths containing the segmentation
- all_keys - list of int, key value that represents positive class
"""
def calculate_dice(all_file_paths, all_keys):
    assert len(all_file_paths) == len(all_keys), "Length of keys does not match number of file paths"
    scores = np.zeros(len(all_file_paths))

    tumormask1 = nib.load(all_file_paths[0]).get_data()
    tumormask2 = nib.load(all_file_paths[1]).get_data()

    if len(tumormask1.shape) == 4:
            tumormask1 = tumormask1[:,:,:,0]
    elif len(tumormask1.shape) > 4:
        raise AssertionError("check shape of " + str(fp))
    if len(tumormask2.shape) == 4:
            tumormask2 = tumormask2[:,:,:,0]
    elif len(tumormask2.shape) > 4:
        raise AssertionError("check shape of " + str(fp))
    tumormask1 = tumormask1.round()
    tumormask2 = tumormask2.round()

    binary_mask1 = (tumormask1 == all_keys[0]) * 1
    binary_mask2 = (tumormask2 == all_keys[1]) * 1

    return calculate_dice_as_np_arrays(binary_mask1, binary_mask2)

"""
Return df with the dice scores of each patient for every label

Inputs:
- base_dir: String, file path of the base directory
- all_labels: list, element represents the name of the mask
- all_keys: list of int, key value that represents positive class
- random_sample: None, float, represents the proportion to randomly sample
  - if None, calculate for all patients
- random_seed: Int, represents the seed to create same random sample
"""
def dice_to_df(base_dir, all_labels, all_keys, random_sample = None, random_seed = None):
    #get all of the file paths of the images to get rapno
    id_file_paths = get_file_paths_of_labels(base_dir, all_labels, random_sample=random_sample, random_seed = random_seed)
    all_ids = id_file_paths[:, 0]
    file_paths = id_file_paths[:, 1]
    

    num_cores = joblib.cpu_count()
    parallel_calculate_dice = partial(calculate_dice, all_keys = all_keys)

    # to allow parallel processing
    with tqdm_joblib(tqdm(desc="Batch Dice scores...", total=len(id_file_paths))) as progress_bar:
        dice_as_list = Parallel(n_jobs=num_cores)(delayed(parallel_calculate_dice)(fps) for fps in file_paths)
    dice_as_list = np.array(dice_as_list).reshape((len(all_ids), 1))

    all_ids = np.array(all_ids).reshape((len(all_ids), 1))
    dice_w_patient_id_as_array = np.hstack((all_ids, np.array(dice_as_list)))

    column_labels = ['Patient_id', 'Dice']
    dice_df = pd.DataFrame(data=dice_w_patient_id_as_array, columns = column_labels)
    dice_df.sort_values(by=['Patient_id'])

    print("Mean Dice Score: ", np.mean(dice_df['Dice']))

    return dice_df

"""
Write csv with the Dice scores of each patient for every label

Inputs:
- base_dir: String, file path of the base directory
- export_to: String, file path of the export_to directory
- all_labels: list, element represents the name of the mask
- all_keys: list of int, key value that represents positive class
- random_sample: None, float, represents the proportion to randomly sample
  - if None, calculate for all patients
- random_seed: Int, represents the seed to create same random sample

Returns:
path of the csv file
"""
def dice_to_csv(base_dir, export_to, all_labels, all_keys, random_sample = None, random_seed = None):
    os.makedirs(export_to, exist_ok=True) #make directory if it does not exist 
    dice_df = dice_to_df(base_dir, all_labels, all_keys, random_sample = random_sample, random_seed = random_seed)
    csv_fp = os.path.join(export_to,  "dice_scores.csv")
    dice_df.to_csv(csv_fp)
    return csv_fp

"""
Return list of volumes that correlate with each file path in inputs

Inputs:
- id_file_path_list - list of 2 elements. First represents the patient ID. 
                        Second is a list with the file paths
- all_keys - list of int, key value that represents positive class
"""
def get_volume_multiple_segmentation(id_file_path_list, all_keys):
    assert len(id_file_path_list[1]) == len(all_keys), "Length of keys does not match number of file paths"
    volumes = []

    for i, (fp, key) in enumerate(zip(id_file_path_list[1], all_keys)):
        tumor_img = nib.load(fp)
        tumormask = tumor_img.get_data()
        voxel_spacing = tumor_img.header.get_zooms()

        if len(tumormask.shape) == 4:
            tumormask = tumormask[:,:,:,0]
        elif len(tumormask.shape) > 4:
            raise AssertionError("check shape of " + str(fp))
        if len(voxel_spacing) == 4:
            voxel_spacing = voxel_spacing[:3]
        elif len(voxel_spacing) > 4:
            raise AssertionError("check header of " + str(fp))
        tumormask = tumormask.round()

        binary_mask = (tumormask == key) * 1

        nvoxels = np.sum(binary_mask)
        voxel_volume = np.prod(voxel_spacing)
        volumes.append(nvoxels * voxel_volume)
    volumes.insert(0, id_file_path_list[0])
    return volumes

"""
Return df with the volumes of each patient for every label

Inputs:
- base_dir: String, file path of the base directory
- all_labels: list, element represents the name of the mask
- all_keys: list of int, key value that represents positive class
- random_sample: None, float, represents the proportion to randomly sample
  - if None, calculate for all patients
- random_seed: Int, represents the seed to create same random sample
"""
def volumes_to_df(base_dir, all_labels, all_keys, random_sample = None, random_seed = None):
    #get all of the file paths of the images to get rapno
    id_file_paths = get_file_paths_of_labels(base_dir, all_labels, random_sample=random_sample, random_seed = random_seed)
    id_file_path_list = id_file_paths.tolist()

    num_processes = multiprocessing.cpu_count()
    parallel_get_volume = partial(get_volume_multiple_segmentation, all_keys = all_keys)

    pool = multiprocessing.Pool(processes=num_processes)
    volume_as_list = []
    for result in tqdm(pool.imap_unordered(func=parallel_get_volume, iterable=id_file_path_list), desc="Batch Volume Scores...", total=len(id_file_path_list)):
        volume_as_list.append(result)

    volume_as_np_array = np.array(volume_as_list)

    column_labels = ['Patient_id']
    column_labels.extend(all_labels)
    volume_df = pd.DataFrame(data=volume_as_np_array, columns = column_labels)
    volume_df.sort_values(by=['Patient_id'])

    return volume_df

"""
Write csv with the volume scores of each patient for every label

Inputs:
- base_dir: String, file path of the base directory
- export_to: String, file path of the export_to directory
- all_labels: list, element represents the name of the mask
- all_keys: list of int, key value that represents positive class
- random_sample: None, float, represents the proportion to randomly sample
  - if None, calculate for all patients
- random_seed: Int, represents the seed to create same random sample

Returns:
path of the csv file
"""
def volume_to_csv(base_dir, export_to, all_labels, all_keys, random_sample = None, random_seed = None):
    os.makedirs(export_to, exist_ok=True) #make directory if it does not exist 
    volume_df = volumes_to_df(base_dir, all_labels, all_keys, random_sample = random_sample, random_seed = random_seed)
    csv_fp = os.path.join(export_to,  "volume_scores.csv")
    volume_df.to_csv(csv_fp)
    return csv_fp

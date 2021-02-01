from extract_masks import rapno_to_csv, dice_to_csv, volume_to_csv, rapno_to_csv_all_lesions

#list of the base directories where the data is
base_dir = [
        "/home/ubuntu/data"
        ]

#list of the export directories where the results will go, make sure it matches order of 'base_dir'
export_to = [
        "/home/ubuntu/metrics"
        ]

#the names of the segmentations of interest
all_labels = [
        ['t1ce_label_RAI_RESAMPLED_BINARY-label.nii.gz', 't1ce_prediction.nii.gz']
        ]

all_keys = [1,1] #the value of the tumor label
random_sample = None #proportion from which to randomly sample the data; pick None or value from 0-1
random_seed = None #pick None or an integer
num_lesions = 4 # the number of lesions to sum, pick 4 for MBL RAPNO criteria

for bd, et, al in zip(base_dir, export_to, all_labels):
        dice_to_csv(bd, et, al, all_keys, random_sample =  random_sample , random_seed = random_seed)
        volume_to_csv(bd, et, al, all_keys, random_sample = random_sample, random_seed = random_seed)
        rapno_to_csv(bd, et, al, all_keys, num_lesions, random_sample = random_sample, random_seed = random_seed)

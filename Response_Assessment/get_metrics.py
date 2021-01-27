from extract_masks import rapno_to_csv, dice_to_csv, volume_to_csv, rapno_to_csv_all_lesions

base_dir = [
        "/home/ubuntu/Documents/data/post_op_predictions/peng/t1ce_output/peng_final_model_t1ce_w_og_dim"
        ]


export_to = [
        "/home/ubuntu/Documents/metrics/postop/peng_t1ce/4lesions_redo"
        ]

all_labels = [
        ['t1ce_label.nii', 't1ce_prediction_RAI_RESAMPLED.nii.gz'],
        ]

all_keys = [1,1]

random_sample = 0.01 #pick None or value from 0-1
random_seed = 0 #pick None or an integer

num_lesions = 4

for bd, et, al in zip(base_dir, export_to, all_labels):
    #rapno_to_csv(bd, et, al, all_keys, num_lesions, random_sample = random_sample, random_seed = random_seed)

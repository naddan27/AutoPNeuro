# AutoPNeuro: Response Assessment


This framework allows for automatic response assessment and calculation of model performance. Framework includes AutoRAPNO, volume, and dice score calculation.

## Notes
Code was developed using Ubuntu 20.04. Code should work on all Linux-based systems.

## AutoRAPNO
AutoRAPNO algorithm is in the 'calculate_RAPNO' script. Images should be in Nifti format for proper reading.

## Response Assessment Pipeline
Use 'get_metrics.py' to get the dice scores, volume, and RAPNO scores of multiple segmentations. Images must be organized in folders by patient at a specific timepoint. In each patient_timepoint folder, all sequences and segmentations should be present with matching names. Example is shown below

--Patient1_timepoint1<br>
----flair.nii.gz<br>
----t1ce.nii.gz<br>
----flair_label.nii.gz<br>
--Patient1_timepoint2<br>
----flair.nii.gz<br>
----t1ce.nii.gz<br>
----flair_label.nii.gz<br>
--Patient2_timepoint1<br>
----flair.nii.gz<br>
----t1ce.nii.gz<br>
----flair_label.nii.gz
# AutoPNeuro

AutoPNeuro is a deep learning Python based framework for automatic segmentation and response assessment for pediatric high grade glioma, medulloblastoma, and other leptomeningeal seeding tumors. The framework consists of image preprocessing, model training and predicting, image postprocessing, and response assessment components.

| Table of Contents |
| ----------------- |
|[1. About](#about) |
|[2. AWS Setup](#aws-setup)|
|[3. Data Organization](#data-organization)|
|[4. Installation](#installation)|
|[5. Image Preprocessing](#image-preprocessing)|
|[6. Model Training](#model-training)|
|[7. Model Predicting](#model-predicting)|
|[8. Utilizing Pre-trained Models](#utilizing-pre-trained-models)|
| [-- T2 Hyperintensity Preoperative](#t2-hyperintensity-segmentation-for-preoperative-cohort)|
| [-- Contrast Enhancing Tumor Preoperative](#contrast-enhancing-tumor-segmentation-for-preoperative-cohort)|
| [-- FLAIR Hyperintensity Preoperative](#flair-hyperintensity-segmentation-for-postoperative-cohort)|
| [-- Contrast Enhancing Tumor Preoperative](#contrast-enhancing-tumor-segmentation-for-postoperative-cohort)|
|[9. Postprocessing](#postprocessing)|
|[10. Response Assessment](#response-assessment)|
|[11. Contact](#contact)|
|[12. Acknowledgements](#acknowledgements)|

## About
AutoPNeuro is a open-sourced deep learning package for pediatric brain tumor segmentation and response assessment developed in Brown University/Rhode Island Hospital Artifical Intelligence Lab. The model training component is based off of DeepNeuro v2. Code is for Linux based systems. Components include:
* Docker containers for data preprocessing, model training, and data postprocessing
* Pre-trained models for preoperative contrast-enhancing tumor, preoperative T2 hyperintensity, postoperative contrast-enhancing tumor, and postoperative FLAIR hyperintensity segmentation
* models are stored in Docker containers for instant deployment 
* tools for model performance evaluation, including volume and 2D measurement comparison

AutoPNeuro is under active development and may be periodically updated.

## AWS Setup
#### Launch GPU Instance
1. Go to AWS Console
2. Click Launch Instance
3. Select *Ubuntu Server 20.04 LTS (HVM), SSD Volume Type 64-bit (x86)* image
4. We will now pick an instance with GPU resources. Instances refer to a predefined number of resources defined by Amazon. The GPU instances start with "p". For most purposes, p3.2x large will be fine. This will give 8vCPUs, 61GB of RAM, and 16GB of GPU memory. To do this, filter the instance family by p3. Then select p3.2xlarge type
![AWS Instance](readme_images/aws_1.png?raw=true)
Note that some regions do not have p family instances. If you can't find the p instance, then switch your region until you find the instance.
5. Pick the amount of storage that you need. You can always upgrade the storage later
6. Configure your security settings
7. Launch your instance. This will prompt you to associate your instance with a key pair. If you are a new user, you can create your own key instance. 
8. AWS will now setup your instance. You should see the status of the instance change to pending. 
9. SSH into your instance

## Data Organization
Data must be separated into 'Train', 'Val', 'Test' folders. In each split, images must be organized in folders by patient at a specific timepoint. In each patient_timepoint folder, all sequences and segmentations should be present with matching names. Example is shown below. Note the example is shown if the model requires 2 different sequences: FLAIR and contrast enhanced T1 sequences

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

## Installation
##### Install NVIDIA Driver
```bash
$ sudo apt-get update
$ sudo apt install ubuntu-drivers-common
$ ubuntu-drivers devices
$ sudo ubuntu-drivers autoinstall
$ sudo reboot
```
You will have to log back into your instance after this

##### Install Docker
```bash
$ sudo apt-get update
$ sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common
$ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
$ sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
$ sudo apt-get update
$ sudo apt-get install docker-ce docker-ce-cli containerd.io
$ sudo docker run hello-world
```

##### Setup NVIDIA Runtime
```bash
$ curl -s -L https://nvidia.github.io/nvidia-container-runtime/gpgkey | \
  sudo apt-key add -
$ distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
$ curl -s -L https://nvidia.github.io/nvidia-container-runtime/$distribution/nvidia-container-runtime.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list
$ sudo apt-get update
$ sudo apt-get install nvidia-container-runtime
$ sudo mkdir -p /etc/systemd/system/docker.service.d
$ sudo tee /etc/systemd/system/docker.service.d/override.conf <<EOF
[Service]
ExecStart=
ExecStart=/usr/bin/dockerd --host=fd:// --add-runtime=nvidia=/usr/bin/nvidia-container-runtime
EOF
$ sudo systemctl daemon-reload
$ sudo systemctl restart docker
$ sudo usermod -aG docker your-user-name
$ sudo reboot
```
SSH back into your instance.

##### Download the training docker
```bash
docker pull dannyhki/rapno_prediction_model:training
docker run --runtime=nvidia -it -v <path_to_data>:/home/neural_network_code/Data/Patients/ dannyhki/rapno_prediction_model:training
```

## Image Preprocessing
Within the training docker container, the preprocessing file to configure is at path:
```
/home/neural_network_code/Code/preprocessing_final.py
```

You will have to change the *vols_to_process* and the *rois_to_process* parameters to the appropriate names. *vols_to_process* represents the names of the images, while *rois_to_process* represents the name of the ground truth segmentations. If preprocessing images on pretrained models, *rois_to_process* can be left as an empty array.

After configuration, run
```
$ python preprocessing_final.py
```

Once preprocessed, images should have the suffix of '_RAI_RESAMPLED_REG_N4_SS_NORM.nii.gz' and segmentations should have the suffix of '_RAI_RESAMPLED_BINARY-label.nii.gz'.

The preprocessing script reorients images to right, anterior, inferior orientation, resamples to isotropic resolution, co-registers the sequences to the same anatomical template, applies N4 Bias correction, skull strips the brain, and normalizes the images. The script also reorients the segmentations to right, anterior, inferior orientation, resamples to isotropic resolution, and binarizes the segmentations.

## Model Training
Once your images have been preprocessed, you may begin model training by first configuring the *config.py* script.

Change *input_image_names* to the names of the preprocessed images and *ground_truth_label_names* to the names of the preprocessed segmentations.  

Make sure that *trainModel* is set to True. Configure other settings to user preferences.

Run preprocessing script
```bash
$ python config.py
```

In practice, ~300 epochs should be able to achieve a model with good performance.

## Model Predicting
Once model has finished training, you may predict with the trained model by changing *trainModel* to False in the *config.py* script.

To predict, run the config file again
```bash
$ python config.py
```

## Utilizing Pre-trained Models
Pre-trained models are available as separate docker containers with the models loaded inside.

For preoperative contrast-enhancing region segmentation:
```
docker pull dannyhki/rapno_prediction_model:preop_t1ce
```
For preoperative T2 hyperintensity segmentation:
```
docker pull dannyhki/rapno_prediction_model:preop_t2
```
For postoperative contrast-enhancing region segmentation:
```
docker pull dannyhki/rapno_prediction_model:postop_t1ce
```
For postoperative FLAIR hyperintensity segmentation:
```
docker pull dannyhki/rapno_prediction_model:postop_flair
```

To use pre-trained models, first follow the Data Organization, Installation, and Image Preprocessing Sections using the training docker. 

Then you can setup one of these pre-trained models onto the preprocessed data. For example, for preoperative contrast enhancing segmentation:
```bash
docker run --runtime=nvidia -it -v <path_to_data>:/home/neural_network_code/Data/Patients/ dannyhki/rapno_prediction_model:preop_t1ce
```

Run the config file to predict using pre-trained models.
```
$ python config.py
```

#### T2 Hyperintensity Segmentation for Preoperative Cohort
This model accepts 1 sequence: T2-weighted images. Rename all T2-weighted images to t2.nii.gz so that the model can find the images.

#### Contrast Enhancing Tumor Segmentation for Preoperative Cohort
This model accepts 2 sequences: contrast-enhanced T1-weighted images and T2-weighted images. Rename all contrast-enhanced T1-weighted images to t1ce.nii.gz and T2-weighted images to t2.nii.gz

#### FLAIR Hyperintensity Segmentation for Postoperative Cohort
This model accepts 2 sequences: contrast-enhanced T1-weighted images and T2-FLAIR images. Rename all contrast-enhanced T1-weighted images to t1ce.nii.gz and FLAIR images to flair.nii.gz

#### Contrast Enhancing Tumor Segmentation for Postoperative Cohort
This model accepts 3 sequences: contrast-enhanced T1-weighted images, T2-FLAIR images, and contrast non-enhanced T1-weighted images. Rename all contrast-enhanced T1-weifhted images to t1ce.nii.gz, FLAIR images to flair.nii.gz, and contrast non-enhanced T1-weighted images to t1.nii.gz

Note that these pretrained models assume that the images have already been preprocessed

## Response Assessment
Response assessment instrunctions can be found [here](https://github.com/naddan27/AutoPNeuro/tree/master/Response_Assessment/README.md). Volumetric response assessment and automatic 2D measurements based on Response Assessment in Pediatric Neuro-Oncology (RAPNO) for medulloblastoma and other leptomeningeal seeding tumors are provided. 

## Contact
AutoPNeuro is under active development. Please send questions to our corresponding author harrison_bai@brown.edu.

## Acknowledgements
AutoPNeuro was built on the DeepNeuro v2 framework. This work would not have been possible without Jay Patel, the primary maintainer of DeepNeuro, and his works. The pretrained models were only possible with the help of our collaborators and their efforts in gathering pediatric brain MRI images. 
Obtaining and Analysing Key Information from Sports Games Using Deep Learning Methods
This is my MSc Dissertation project ([link](https://drive.google.com/file/d/1n12fraqAJDZC6uTyVHHqIV-s5bRIlO2R/view?usp=sharing)): 
developed methods to get the data from tracked objects for further statistical analysis. This required to train a detection
model (YOLOv5) on a custom dataset and modifying the tracking algorithm (StrongSORT) to perform implemented in the 
project features. A project involved detection and tracking models and was developed in Jupyter Notebook using PyTorch.

[YouTube](https://www.youtube.com/watch?v=7RarBqKHYn0) link









///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Instructions on how to run the code.

The track.py file is a part of the StrongSORT tracking model and was modified for the purpose of this project 
to perform all implemented features by implementing count_obj(), motion_path() and stat_data() functions.

All instructions in more detail are described in the main_code.ipynb file.

1.	Download the following files main_code.ipynb, custom_data.yaml, Tennis_V4.zip, track.py, and extract if needed.
2.	From main_code.ipynb can extract the Tennis_V4.zip dataset into a specified directory.
3.	Clone the GitHub repository from ultralytics, consisting of YOLOv5 models from this link, 
	and install all necessary requirements from the requirements.txt file inside that directory.
4.	To train the model on a custom dataset from this project, need to update the dataset directory path in custom_data.yaml file if necessary 
	and upload that file into “ ’YOUR PROJECT DIRECTORY’/yolov5/data/coco128.yaml”.
5.	To train a specific model with the usage of data augmentation from albumentations need to install it in the same directory by using these commands:
	!pip install -U albumentations 
	!echo "$(pip freeze | grep albumentations) "
6.	To perform training on a specific YOLOv5 model, follow the instructions and commands inside main_code.ipynb, 
	which show how to specify hyperparameters, image input size, optimizer, the number of epochs, batch size, weights, the model architecture, and a few more.
7.	To perform tracking, first, need to clone the GitHub repository with the StrongSORT model from this link.
8.	After cloning the repository, need to install all requirements from the requirements.txt file inside that directory.
9.	All proposed in this project implementations for a tracking model are inside the track.py file. 
	Need to replace this file with its original counterpart inside the StrongSORT directory from cloned repository.
10.	Inside main_code.ipynb code file, follow instructions under the “Tracking” section, where presented how to specify paths to input video file, 
	weights from pre-trained YOLOv5 detection model, activate only specific object class if needed, confidence threshold or input video size for tracking model.
11.	Further processing of extracted and saved data from tracked objects is located in main_code.ipynb file, under the “Further Processing - Extracting Data”.

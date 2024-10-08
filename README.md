Research on the SEGDC-UNet Electron Microscope Image Segmentation Algorithm Based on Channel Attention Mechanism

This repository contains the implementation of SEGDC-UNet, a novel deep learning model designed for electron microscope (EM) image segmentation tasks. The SEGDC-UNet architecture incorporates a channel attention mechanism and the GELU activation function into the DC-UNet backbone, enhancing the model's ability to focus on essential features and improving segmentation performance.

![image](https://github.com/octlib/li/assets/141291477/c8c0e866-7358-4925-9c7b-70bc6cfaf76c)


The EMPS-Augmented dataset can be found at https://www.kaggle.com/datasets/liyue123/emps-augmented-pixel.

To reference the SEGDC_Unet from the SEGDC file in your main.py, and set in_channels to 1 for training, follow these steps:

1. Import SEGDC_Unet Model
First, make sure to properly import the SEGDC_Unet model in main.py. Assuming the SEGDC_Unet.py file is located in the models folder, use the following code to import the model:
from models.SEGDC import SEGDC_Unet
2. Set in_channels=1
When initializing the model, you can set the in_channels parameter to 1 like this:
Set in_channels to 1 during model initialization
model = SEGDC_Unet(in_channels=1, out_channels=1)  # Adjust out_channels as needed


test.py ：It is the test set data, specifically a series of filenames in the `emps/test.csv` file. 

The functionality of `test.py` is as follows: 
1. Load the `best_model.pth.tar` model; 
2. Load the test set data; 
3. Run the model on all images in the test set and output the average Accuracy, Dice score, Recall, and IoU.
4. The weight file and test dataset can be accessed at https://pan.baidu.com/s/1q4mRJUFz3OFk4V05Uf2X0A 
extract code：hxld

Acknowledgments
We thank the contributors to the open-source projects that made this work possible, including PyTorch, and the EMPS dataset providers. Special thanks to the reviewers for their invaluable feedback.


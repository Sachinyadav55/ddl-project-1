# DDL-Project
Team 09 DDL project - Manu, Vineeth, Priyam, Sachin

Course webpage : https://www.cc.gatech.edu/~jarulraj/courses/8803-f19/

Our current project consists of these tasks:
1. Using openCV to detect motion and remove empty frames
2. Using noscope to remove empty frames using light weight neural network
3. Training 3D-CNN, performing hyperparameter tuning
4. Tweaking the network 
5. Comparing the metrics with pretrained networks of PyTorch trained on Kinetics dataset and trying out different architectures
6. Integration with EVA

### Installing Dependencies
Make sure to install following dependencies before running the code

```bash
	pip install -r requirements.txt
   ```

#### For noscope:
follow the installation instructions from here: https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html
### Using OpenCV

Once the dependencies are installed, the frames associated with motion can be obtained by using following command

```bash
	python3 opencvFilter.py -v /path/to/any/video/ [-a mimimum_area_threshold]
```
   
  This will create a CSV with the same name as video at the same location. The CSV should contain two columns of frameID and Include/Exclude boolean.
  

### Using 3d-CNN
Please read the README.md for 3D-ResNets-PyTorch for implimentation of 3d-CNN

Credits:https://github.com/kenshohara/3D-ResNets-PyTorch


```bibtex
@inproceedings{hara3dcnns,
  author={Kensho Hara and Hirokatsu Kataoka and Yutaka Satoh},
  title={Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet?},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={6546--6555},
  year={2018},
}
```


### Hyperparameter Tuning

Once the 3d-ResNet-PyTorch is tested, the statistics for hyperparameters can be obtained by following command

```bash
	anaconda3/bin/python 3D-ResNets-PyTorch/hyperparameterTuning.py
```
This will perform Grid search on the models, modelDepths, learning_rates, weight_decays, and optimizers.

The script will generate "hyperparameterTuningLog.csv" for saving runtimes and "log.txt" for saving running logs for each iteration.

### Fine tuning and testing pretrained models

We have tweaked the final layer of pretrained model for video classification in PyTorch trained on Kinetic-400 dataset. Refer to "PyTorch_3dCNN-Testing.ipynb" for walk through.


### Using main.py for video classification

Make sure you have a GPU configured for training the model.

### Running the script:


```bash
	python3 model.py
```

This will train the model with default metrics (refer the model.py) and generate results in directory containing saved model(s) and confusion matrix. To save the log information redirect the output to logFile.txt

By default all GPUs is used for the training.
If you want a part of GPUs, set devices using ```CUDA_VISIBLE_DEVICES=...```.

Please go through the list of arguments which parser uses and set the arguments accordingly.

### Using Difference Detector and Image Labeling Scripts

Please refer to the README.md in the respective folders to run the script.


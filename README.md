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

### Using OpenCV

Once the dependencies are installed, the frames associated with motion can be obtained by using following command

```bash
	python3 opencvFilter.py -v /path/to/any/video/ [-a mimimum_area_threshold]
```
   
  This will create a CSV with the same name as video at the same location. The CSV should contain two columns of frameID and Include/Exclude boolean.
  
### Using Noscope



### Using 3d-CNN
Please read the README.md for 3D-ResNets-PyTorch for implimentation of 3d-CNN

### Hyperparameter Tuning

Once the 3d-ResNet-PyTorch is tested, the statistics for hyperparameters can be obtained by following command

```bash
	anaconda3/bin/python 3D-ResNets-PyTorch/hyperparameterTuning.py
```
This will perform Grid search on the models, modelDepths, learning_rates, weight_decays, and optimizers.

The script will generate "hyperparameterTuningLog.csv" for saving runtimes and "log.txt" for saving running logs for each iteration.

### Fine tuning and testing pretrained models

We have tweaked the final layer of pretrained model for video classification in PyTorch trained on Kinetic-400 dataset. Refer to "PyTorch_3dCNN-Testing.ipynb" for walk through.
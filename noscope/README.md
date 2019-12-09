# This code is for using naive noscope implementation

Implemented by Manu Arrojwala and this won't be integrated to EVA as it is redundant
#### The code has following functionalities:
1. Cascade of shallow neural networks, with an option to try out different models
2. Customized object detection inference using a tensorflow. The underlying network can be pre-trained as the ones in the tensorflow zoo or you can train one from scratch (more info at https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html)

*object_detect.py*: modified inference script for tensorflow 1.12 checkpoints (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). The GPU allocation for the network has been tweaked to decrease the inference time (from 3.5~4fps to ~0.25 fps). In a nutshell, this script runs the full model on 10000 random frames from the video. These frames will be used to train the subsequent shallow neural nets

*cascade.py*: deprecated file used to check if the alexnet and alexnet-like models can work as efficient shallow neural networks with near full model accuracy (spoiler alert: it didn't work!)

*cascade-pretrained.py*: same overall functionalityy of cascade.py but with an option to use pretrained networks (VGG, inception, ResNet, Squeezenet). Also has the functionality to run the trained shallow neural net on the whole video. (speed = ~125fps)


import os, subprocess


# time python main.py --root_path ~/data --video_path kinetics_videos/jpg --annotation_path kinetics.json 
# --result_path results --dataset kinetics --model resnet --model_depth 10 --n_classes 2 --batch_size 1 --n_threads 4 --checkpoint 50

models = ['resnet', 'densenet', 'pre_act_resnet', 'resnext', 'wide_resnet']
modelDepths = ['10', '18', '34']

parameters = {}
parameters['root_path'] = '/data/home/valjapur3/data'
parameters['video_path'] = 'kinetics_videos/jpg'
parameters['annotation_path'] = 'kinetics.json'
parameters['result_path'] = 'results'
parameters['dataset'] = 'kinetics'
parameters['model'] = 'resnet'
parameters['model_depth'] = '10'
parameters['n_classes'] = '2'
parameters['batch_size'] = '1'
parameters['n_threads'] = '4'
parameters['checkpoint'] = '50'

cmd =  ['/data/home/valjapur3/anaconda3/bin/python', 'main.py', '--root_path', parameters['root_path'], '--video_path', parameters['video_path'],
		'--annotation_path', parameters['annotation_path'], '--result_path', parameters['result_path'],
		'--dataset', parameters['dataset'], '--model', parameters['model'], '--model_depth', parameters['model_depth'], 
		'--n_classes', parameters['n_classes'], '--batch_size', parameters['batch_size'], 
		'--n_threads', parameters['n_threads'], '--checkpoint', parameters['checkpoint']]

with open(os.path.join(parameters['root_path'], 'stdout'+ parameters['model']+parameters['model_depth']), 'w') as f:
	subprocess.call(cmd, stdout=f)
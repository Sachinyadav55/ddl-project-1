#!/usr/bin/env python
# coding: utf-8

import os, subprocess, time

models = ['resnet', 'densenet', 'pre_act_resnet', 'resnext', 'wide_resnet']
modelDepths = ['10', '18', '34']
learning_rates = ['0.1', '0.01', '0.8']
weight_decays = ['1e-3', '1e-2', '1e-4']
# momentums = ['0.9', '0.5', '0.1']
optimizers = ['sgd', 'Adam']

parameters = {}
parameters['root_path'] = os.path.join(os.environ['HOME'], 'data')
parameters['video_path'] = 'kinetics_videos/jpg'
parameters['annotation_path'] = 'kinetics.json'
parameters['result_path'] = 'results'
parameters['dataset'] = 'kinetics'
parameters['n_classes'] = '2'
parameters['batch_size'] = '1'
parameters['n_threads'] = '4'
parameters['checkpoint'] = '50'

parameters['model'] = 'resnet'

parameters['model_depth'] = '10'
parameters['learning_rate'] = '0.1'
parameters['momentum'] = '0.9'
parameters['weight_decay'] = '1e-3'

pythonPath = os.path.join(os.environ['HOME'], 'anaconda3/bin/python')

#Grid search for all the hyper parameters
with open(os.path.join(parameters['root_path'], 'hyperparameterTuningLog.csv'), 'w') as f1: #For saving runtimes
	f1.write('model,runtime')
	for model in models:
		for modelDepth in modelDepths:
			for learning_rate in learning_rates:
				for weight_decay in weight_decays:
					for optimizer in optimizers:

						results = '_'.join([model, modelDepth, learning_rate, weight_decay, weight_decay, optimizer])
						print ('running', results)
						resultsPath = os.path.join(parameters['root_path'], results)

						if not os.path.exists(resultsPath):
							os.makedirs(resultsPath)

						cmd =  [pythonPath, 'main.py', '--root_path', 
						parameters['root_path'], '--video_path', parameters['video_path'], 
						'--annotation_path', parameters['annotation_path'], '--result_path', results,
						'--dataset', parameters['dataset'], '--model', model, '--model_depth', parameters['model_depth'], 
						'--n_classes', parameters['n_classes'], '--batch_size', parameters['batch_size'], 
						'--n_threads', parameters['n_threads'], '--checkpoint', parameters['checkpoint'], 
						'--learning_rate', learning_rate, '--weight_decay', weight_decay, '--optimizer', optimizer]

						try:
							t1 = time.time()

							with open(os.path.join(resultsPath, 'log.txt'), 'w') as f2: #For saving running logs
								subprocess.call(cmd, stdout=f2)

							t2 = time.time()

							log = results + ',' + str(t2 - t1)

							f1.write(log)

						except:
							log = results + ',' + 'Failed'

							f1.write(log)

print('Done!')
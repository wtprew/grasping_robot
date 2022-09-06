import os
import sys

import numpy as np
import torch

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

class Logger(object):
	def __init__(self, file):
		self.terminal = sys.stdout
		self.log = open(file, "a")

	def write(self, message):
		self.terminal.write(message)
		self.log.write(message)  

	def flush(self):
		pass    

def perf_measure(y_actual, y_hat):
	TP = 0
	FP = 0
	TN = 0
	FN = 0
	for i in range(len(y_hat)): 
		if y_actual[i]==y_hat[i]==1:
		   TP += 1
		if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
		   FP += 1
		if y_actual[i]==y_hat[i]==0:
		   TN += 1
		if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
		   FN += 1
	tp = TP/len(y_actual)
	fp = FP/len(y_actual)
	tn = TN/len(y_actual)
	fn = FN/len(y_actual)
	return(TP, FP, TN, FN), (tp, fp, tn, fn)

def save_checkpoint(state, is_best, checkpoint):
	"""Saves model and training parameters at checkpoint + 'last.pth.tar'
   	Args:
		state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
		checkpoint: (string) folder where parameters are to be saved
   	"""
	filepath = os.path.join(checkpoint, 'last.pth.tar')
	bestpath = os.path.join(checkpoint, 'best.pth.tar')
	if is_best:
		print ("=> Saving a new best")
		torch.save(state, bestpath)  # save checkpoint
	if not os.path.exists(checkpoint):
		print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
		os.mkdir(checkpoint)
	else:
		print("Checkpoint Directory exists! ")
	torch.save(state, filepath)

class VisdomLinePlotter(object):
	"""Plots to Visdom"""
	def __init__(self, env_name='main', server='http://rrhk62@ncc.clients.dur.ac.uk', port=2506):
		self.viz = Visdom(server=server,port=port)
		self.env = env_name
		self.plots = {}

	def batchplot(self, var_name, split_name, title_name, x, y):
		if var_name not in self.plots:
			self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
				legend=[split_name],
				title=title_name,
				xlabel='Batch',
				ylabel=var_name
			))
		else:
			self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')
	
	def epochplot(self, var_name, split_name, title_name, x, y):
		if var_name not in self.plots:
			self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
				legend=[split_name],
				title=title_name,
				xlabel='Epoch',
				ylabel=var_name
			))
		else:
			self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')

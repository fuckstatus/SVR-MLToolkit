#!/usr/bin/env python
# This gridregression.py is modified from grid.py in LIBSVM 3.17 version
__all__ = ['find_parameters']

import os, sys, traceback, getpass, time, re
from threading import Thread
from subprocess import *

if sys.version_info[0] < 3:
	from Queue import Queue
else:
	from queue import Queue

telnet_workers = []
ssh_workers = []
nr_local_worker = 1

class GridOption:
	def __init__(self, dataset_pathname, options):
		dirname = os.path.dirname(__file__)
		if sys.platform != 'win32':
			self.svmtrain_pathname = os.path.join(dirname, '../julia/shellvalidation.jl')
			self.gnuplot_pathname = '/usr/bin/gnuplot'
		else:
			# example for windows
			self.svmtrain_pathname = os.path.join(dirname, r'..\windows\svm-train.exe')
			# svmtrain_pathname = r'c:\Program Files\libsvm\windows\svm-train.exe'
			self.gnuplot_pathname = r'c:\tmp\gnuplot\binary\pgnuplot.exe'
		self.fold = 5
		self.c_begin, self.c_end, self.c_step = -1,  6,  1
		self.g_begin, self.g_end, self.g_step =  0, -8, -1
		self.p_begin, self.p_end, self.p
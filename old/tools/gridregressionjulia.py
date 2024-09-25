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
		self.p_begin, self.p_end, self.p_step = -8,  -1,  1
		self.grid_with_c, self.grid_with_g, self.grid_with_p = True, True, True
		self.dataset_pathname = dataset_pathname
		self.dataset_title = os.path.split(dataset_pathname)[1]
		self.out_pathname = '{0}.out'.format(self.dataset_title)
		self.png_pathname = '{0}.png'.format(self.dataset_title)
		self.pass_through_string = ' '
		self.resume_pathname = None
		self.parse_options(options)

	def parse_options(self, options):
		if type(options) == str:
			options = options.split()
		i = 0
		pass_through_options = []
		
		while i < len(options):
			if options[i] == '-log2c':
				i = i + 1
				if options[i] == 'null':
					self.grid_with_c = False
				else:
					self.c_begin, self.c_end, self.c_step = map(float,options[i].split(','))
			elif options[i] == '-log2g':
				i = i + 1
				if options[i] == 'null':
					self.grid_with_g = False
				else:
					self.g_begin, self.g_end, self.g_step = map(float,options[i].split(','))
			elif options[i] == '-log2p' :
				i = i + 1
				if options[i] == 'null':
					self.grid_with_p = False	
				else:
					self.p_begin, self.p_end, self.p_step = map(float,options[i].split(','))
			elif options[i] == '-v':
				i = i + 1
				self.fold = options[i]
			elif options[i] in ('-c','-g','-p'):
				raise ValueError('Use -log2c , -log2g and -log2p.')
			elif options[i] == '-svmtrain':
				i = i + 1
				self.svmtrain_pathname = options[i]
			elif options[i] == '-gnuplot':
				i = i + 1
				if options[i] == 'null':
					self.gnuplot_pathname = None
				else:	
					self.gnuplot_pathname = options[i]
			elif options[i] == '-out':
				i = i + 1
				if options[i] == 'null':
					self.out_pathname = None
				else:
					self.out_pathname = options[i]
			elif options[i] == '-png':
				i = i + 1
				self.png_pathname = options[i]
			elif options[i] == '-resume':
				if i == (len(options)-1) or options[i+1].startswith('-'):
					self.resume_pathname = self.dataset_title + '.out'
				else:
					i = i + 1
					self.resume_pathname = options[i]
			else:
				pass_through_options.append(options[i])
			i = i + 1

		self.pass_through_string = ' '.join(pass_through_options)
		if not os.path.exists(self.svmtrain_pathname):
			raise IOError('svm-train executable not found')
		if not os.path.exists(self.dataset_pathname):
			raise IOError('dataset not found')
		if self.resume_pathname and not os.path.exists(self.resume_pathname):
			raise IOError('file for resumption not found')
		if not self.grid_with_c and not self.grid_with_g and not self.grid_with_p:
			raise ValueError('-log2c , -log2g and -log2p should not be null simultaneously')
		# gridregression default did not support gnuplot
		self.gnuplot_pathname = None
		if self.gnuplot_pathname and not os.path.exists(self.gnuplot_pathname):
			sys.stderr.write('gnuplot executable not found\n')
			self.gnuplot_pathname = None

def redraw(db,best_param,gnuplot,options,tofile=False):
	if len(db) == 0: return
	begin_level = round(max(x[2] for x in db)) - 3
	step_size = 0.5

	best_log2c,best_log2g,best_rate = best_param

	# if newly obtained c, g, or cv values are the same,
	# then stop redrawing the contour.
	if all(x[0] == db[0][0]  for x in db): return
	if all(x[1] == db[0][1]  for x in db): return
	if all(x[2] == db[0][2]  for x in db): return

	if tofile:
		gnuplot.write(b"set term png transparent small linewidth 2 medium enhanced\n")
		gnuplot.write("set output \"{0}\"\n".format(options.png_pathname.replace('\\','\\\\')).encode())
		#gnuplot.write(b"set term postscript color solid\n")
		#gnuplot.write("set output \"{0}.ps\"\n".format(options.dataset_title).encode().encode())
	elif sys.platform == 'win32':
		gnuplot.write(b"set term windows\n")
	else:
		gnuplot.write( b"set term x11\n")
	gnuplot.write(b"set xlabel \"log2(C)\"\n")
	gnuplot.write(b"set ylabel \"log2(gamma)\"\n")
	gnuplot.write("set xrange [{0}:{1}]\n".format(options.c_begin,options.c_end).encode())
	gnuplot.write("set yrange [{0}:{1}]\n".format(options.g_begin,options.g_end).encode())
	gnuplot.write(b"set contour\n")
	gnuplot.write("set cntrparam levels incremental {0},{1},100\n".format(begin_level,step_size).encode())
	gnuplot.write(b"unset surface\n")
	gnuplot.write(b"unset ztics\n")
	gnuplot.write(b"set view 0,0\n")
	gnuplot.write("set title \"{0}\"\n".format(options.dataset_title).encode())
	gnuplot.write(b"unset label\n")
	gnuplot.write("set label \"Best log2(C) = {0}  log2(gamma) = {1}  accuracy = {2}%\" \
				  at screen 0.5,0.85 center\n". \
				  format(best_log2c, best_log2g, best_rate).encode())
	gnuplot.write("set label \"C = {0}  gamma = {1}\""
				  " at screen 0.5,0.8 center\n".format(2**best_log2c, 2**best_log2g).encode())
	gnuplot.write(b"set key at screen 0.9,0.9\n")
	gnuplot.write(b"splot \"-\" with lines\n")
	
	db.sort(key = lambda x:(x[0], -x[1]))

	prevc = db[0][0]
	for line in db:
		if prevc != line[0]:
			gnuplot.write(b"\n")
			prevc = line[0]
		gnuplot.write("{0[0]} {0[1]} {0[2]}\n".format(line).encode())
	gnuplot.write(b"e\n")
	gnuplot.write(b"\n") # force gnuplot back to prompt when term set failure
	gnuplot.flush()


def calculate_jobs(options):
	
	def range
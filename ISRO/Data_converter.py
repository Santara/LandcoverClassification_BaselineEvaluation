import pickle as pkl
import numpy as np
import random
# the_list = range(100)

def convert_to_one_hot(integer_targets, num_classes):
	num_samples =len(integer_targets)
	one_hot_target_matrix = np.zeros((num_samples, num_classes))
	for i in xrange(num_samples):
		one_hot_target_matrix[i][integer_targets[i]] = 1

	return one_hot_target_matrix

if __name__=='__main__':
	demo_vec = np.arange(16)
	one_hot_target_matrix = convert_to_one_hot(demo_vec,num_classes=16)
	print "Input vector: ", demo_vec
	print "Output: ", one_hot_target_matrix
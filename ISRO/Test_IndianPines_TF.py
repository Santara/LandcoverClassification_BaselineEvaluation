import tensorflow as tf
import numpy as np
import pickle as pkl
import random
from Data_converter import convert_to_one_hot

POOL_SIZE = 5
INPUT_CHANNEL = 220
OUTPUT_CLASSES = 16
KERNEL_1 = 21
FILTERS_1 = 20
HIDDEN_LAYER_1 = 100

f = pkl.load(open('our_data.pkl','rb'))
data = f['data']

""" define test train data"""
def batch(data,k):
	batch = random.sample(data, k)
	inputs =[]
	targets = []
	for a,b in batch:
		inputs.append(a)
		targets.append(b)
	targets = convert_to_one_hot(targets,OUTPUT_CLASSES)
	return inputs, targets


""" define placeholder """
#placeholder
x = tf.placeholder("float",name='x',shape=([None,INPUT_CHANNEL,1,1]))
y = tf.placeholder("float",name='y',shape=([None,OUTPUT_CLASSES]))


""" define function """
def weight_variable(shape):
	init = tf.truncated_normal(shape,stddev=1.0)
	return tf.Variable(init)

def bias_variable(shape):
	init = tf.constant(0.1,shape=shape)
	return tf.Variable(init)

def conv_2d(x,W):
	return tf.nn.conv2d(x,W,strides = [1,1,1,1], padding='VALID')

def max_pool_5x1(x):
	return tf.nn.max_pool(x,ksize=[1,POOL_SIZE,1,1],strides=[1,POOL_SIZE,1,1],padding='SAME') # check k-size


""" making the model """
# shape = [width.height,depth,output size]
w_conv1 = weight_variable([KERNEL_1,1,1,FILTERS_1])
b_conv1 = bias_variable([FILTERS_1])

h_conv1 = tf.nn.relu(conv_2d(x,w_conv1) + b_conv1)
h_pool1 = max_pool_5x1(h_conv1)

# full connected-1
#h_pool1_flat = tf.reshape(h_pool1,[1,-1])
W_fc1 = weight_variable([40*1*20,HIDDEN_LAYER_1])
b_fc1 = bias_variable([HIDDEN_LAYER_1])
h_fc1 = tf.nn.relu(tf.matmul(h_pool1, W_fc1) + b_fc1) 

# droupout layer 
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

# softmax layer
W_fc2 = weight_variable([HIDDEN_LAYER_1,OUTPUT_CLASSES])
b_fc2 = bias_variable([OUTPUT_CLASSES])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)


""" Optimization Implentation """
# define cost function
cross_entropy = -tf.reduce_sum(y*tf.log(y_conv))

# define traiing algorithm (minimize cross entropy)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# define accuracy for prediction
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


""" start session """
# start session
sess = tf.Session()
sess.run(tf.initialize_all_variables())


""" start training """

for i in range(200):
	inp,tar = batch(data,50)
	print len(inp),len(inp[0])
	inp = np.asarray(inp).reshape(50,220,1,1)
	tar = np.asarray(tar)
	print inp.shape, tar.shape

	if i%10 == 0:
		train_accuracy = sess.run( accuracy, feed_dict={x:inp, y: tar, keep_prob: 1.0})
		print("step %d, training accuracy %g"%(i, train_accuracy))
	sess.run(train_step,feed_dict={x: inp, y: tar, keep_prob: 0.5})


inp,tar = batch(data,200)
print("test accuracy %g"% sess.run(accuracy, feed_dict={x: inp, y: tar, keep_prob: 1.0}))

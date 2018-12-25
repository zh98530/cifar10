#coding:utf-8

import tensorflow as tf

#输入图片为24*24*3，第一层卷积核为5*5*3*64，第二层卷积核为5*5*64*64
IMAGE_SIZE = 24
NUM_CHANNELS = 3
CONV1_SIZE = 5
CONV1_KERNEL_NUM = 64
CONV2_SIZE = 5
CONV2_KERNEL_NUM = 64
FC1_SIZE = 384
FC2_SIZE = 192
OUTPUT_NODE = 10


def get_weight(shape, stddev, regularizer):
    w = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if regularizer != None:
        weight_loss = tf.multiply(tf.nn.l2_loss(w), regularizer, name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return w


def get_bias(value, shape):
    b = tf.Variable(tf.constant(value, shape=shape))
    return b

#卷积计算函数提取特征，x为输入，w为卷积核，步长为[横向:1,纵向:1]，padding选择全零填充
def conv2d(x,w):
	return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')

#池化计算函数取max，x为输入，池化核为[3,3]，步长为[横向:2,纵向:2]，padding选择全零填充
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

def forward(x,train,batch_size):
	#get_weight待优化参数conv1_w卷积核5*5,3通道，64核;64个核对应64个偏置conv1_b
	conv1_w = get_weight(shape=[CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_KERNEL_NUM], stddev=5e-2, regularizer=0.0)
	conv1_b = get_bias(0.0, shape=[CONV1_KERNEL_NUM])

	#第一层卷积，并激活，池化
	conv1 = conv2d(x, conv1_w)
	relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_b))
	pool1 = max_pool_2x2(relu1)

	#局部响应归一化处理，增强模型的泛化能力
	norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

	#get_weight待优化参数conv2_w卷积核5*5,32通道，64核;64个核对应64个偏置conv2_b
	weight2 = get_weight(shape=[CONV2_SIZE, CONV2_SIZE, CONV1_KERNEL_NUM, CONV2_KERNEL_NUM], stddev=5e-2, regularizer=0.0)
	conv2_b = get_bias(0.1, shape=[CONV2_KERNEL_NUM])

	#第二层卷积，并激活，归一化，池化
	conv2 = conv2d(norm1, weight2)
	relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))
	norm2 = tf.nn.lrn(relu2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
	pool2 = max_pool_2x2(norm2)

	reshape = tf.reshape(pool2, [batch_size, -1])
	dim = reshape.get_shape()[1].value
    
	fc1_w = get_weight(shape=[dim, FC1_SIZE], stddev=0.04, regularizer=0.004)
	fc1_b = get_bias(0.1, shape=[FC1_SIZE])
	fc1 = tf.nn.relu(tf.matmul(reshape, fc1_w)+fc1_b)
	if train:fc1 = tf.nn.dropout(fc1,0.5)

	fc2_w = get_weight(shape=[FC1_SIZE, FC2_SIZE], stddev=0.04, regularizer=0.004)
	fc2_b = get_bias(0.1, shape=[FC2_SIZE])
	fc2 = tf.nn.relu(tf.matmul(fc1, fc2_w)+fc2_b)

	fc3_w = get_weight(shape=[FC2_SIZE, OUTPUT_NODE], stddev=1 / 192.0, regularizer=0.0)
	fc3_b = get_bias(0.0, shape=[OUTPUT_NODE])
	y = tf.add(tf.matmul(fc2, fc3_w), fc3_b)
	return y

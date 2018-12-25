# coding:utf-8

import cifar10, cifar10_input
import tensorflow as tf
import numpy as np
import time
import cifar10_forward
import os
import math

max_steps = 3000
BATCH_SIZE = 128
data_dir = "./data/"

LEARNING_RATE_BASE = 0.005
MOVING_AVERAGE_DECAY = 0.99
LEARNING_RATE_DECAY = 0.96

MODEL_SAVE_PATH = "./Model/"
MODEL_NAME = "cifar10_model"


def backward(cifar10_image, cifar10_labels):
	images_test,labels_test = cifar10_input.inputs(eval_data=True,data_dir=data_dir,batch_size=BATCH_SIZE)
	
	x = tf.placeholder(tf.float32, [
		BATCH_SIZE,
		cifar10_forward.IMAGE_SIZE,
		cifar10_forward.IMAGE_SIZE,
		cifar10_forward.NUM_CHANNELS])
	y_ = tf.placeholder(tf.int32, [BATCH_SIZE])
	y = cifar10_forward.forward(x, True, BATCH_SIZE)
	global_step = tf.Variable(0, trainable=False)

	def loss(logits, labels):
		labels = tf.cast(labels, tf.int64)
		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross_entropy_per_example')
		cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
		tf.add_to_collection('losses', cross_entropy_mean)
		return tf.add_n(tf.get_collection('losses'), name='total_loss')

	loss = loss(y, y_)

	# 学习率定为指数衰减型
#	learning_rate = tf.train.exponential_decay(
#		LEARNING_RATE_BASE,
#		global_step,
#		60000 / BATCH_SIZE,
#		LEARNING_RATE_DECAY,
#		staircase=True)

#	train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
	train_step = tf.train.AdamOptimizer(1e-3).minimize(loss, global_step=global_step)

	# 滑动平均更新新的神经网络参数，保留上一次的参数影响，给予其合适的权重影响新参数
	ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
	ema_op = ema.apply(tf.trainable_variables())
	with tf.control_dependencies([train_step, ema_op]):
		train_op = tf.no_op(name='train')

	top_k_op = tf.nn.in_top_k(y,y_,1)
	# 实例化saver对象
	saver = tf.train.Saver()

	with tf.Session() as sess:
		init_op = tf.global_variables_initializer()
		sess.run(init_op)

		coord = tf.train.Coordinator()
		tf.train.start_queue_runners(coord=coord)
			
		image_batch,label_batch = sess.run([images_test,labels_test])

		ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
		if ckpt and ckpt.model_checkpoint_path:
			# 加载保存的会话
			saver.restore(sess, ckpt.model_checkpoint_path)

		for step in range(max_steps):
			start_time = time.time()
			image_batch, label_batch = sess.run([cifar10_image, cifar10_labels])
			_, loss_value = sess.run([train_op, loss], feed_dict={x: image_batch, y_: label_batch})
			duration = time.time() - start_time
			if step % 10 == 0:
				examples_per_sec = BATCH_SIZE / duration
				sec_per_batch = float(duration)

				format_str = ('step %d,loss=%.2f (%.1f examples/sec;%.3f sec/batch)')
				print(format_str % (step, loss_value, examples_per_sec, sec_per_batch))
				saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME),global_step=global_step)
				
				
				num_examples = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
				num_iter = int(num_examples / BATCH_SIZE)
				true_count = 0
				total_sample_count = num_iter*BATCH_SIZE
				step = 0
				for step in range(num_iter):
#					image_batch,label_batch = sess.run([images_test,labels_test])
					predictions = sess.run([top_k_op],feed_dict={x:image_batch,y_:label_batch})
					true_count += np.sum(predictions)
#					print('precision @ 1 = %.3f'%(num_iter))

				print('%d,%d' % (true_count,total_sample_count))

def main():
	cifar10_image, cifar10_labels = cifar10_input.distorted_inputs(data_dir=data_dir, batch_size=BATCH_SIZE)
	backward(cifar10_image, cifar10_labels)


if __name__ == '__main__':
	main()

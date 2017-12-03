import tensorflow as tf
import numpy as np
import sklearn as skl
import glob
import os
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from loader import WordLoader

K = 5
epochs = 20
data = WordLoader(data=K, glove=50, resample=True)
folder = '5-hidden'

if not os.path.isdir(folder):
	os.mkdir(folder)

with tf.Session() as sess:
	try:
		print('loading')
		files = glob.glob("%s/*.meta" % folder)
		if not files:
			raise IOError

		files.sort(key=lambda x: -os.path.getmtime(x))
		meta_file = files[0]
		checkpoint_name = meta_file.split('.')[0]

		saver = tf.train.import_meta_graph(meta_file)
		saver.restore(sess, checkpoint_name)

		graph = tf.get_default_graph()
		prediction = graph.get_tensor_by_name('prediction:0')
		global_step = graph.get_tensor_by_name('global_step:0')
		loss = graph.get_tensor_by_name('loss:0')
		keep_rate = graph.get_tensor_by_name('keep_rate:0')
		input_words = graph.get_tensor_by_name('word_ids:0')
		true_emojis = graph.get_tensor_by_name('emoji_ids:0')
		optimizer = graph.get_operation_by_name('optimizer')
	except (IOError, tf.errors.NotFoundError) as e:
		print('failed to load')
		input_words = tf.placeholder(tf.int64, [data.batch_size, data.max_seq_len], name='word_ids')
		word_embeds = tf.get_variable("word_embedding",
				initializer=tf.constant(data.glove_embed, dtype=tf.float32))
		embed_words = tf.nn.embedding_lookup(word_embeds, input_words)

		cnst_init = tf.constant_initializer(0.0)

		kernel1 = tf.get_variable(name='kernel1_weight', shape=[3, 50, 64],	initializer=tf.random_uniform_initializer(
			minval=-6/(data.max_seq_len + 64), maxval=6/(data.max_seq_len + 64)))
		kernel2 = tf.get_variable(name='kernel2_weight', shape=[4, 50, 64], initializer=tf.random_uniform_initializer(
			minval=-8/(data.max_seq_len + 64), maxval=8/(data.max_seq_len + 64)))
		kernel3 = tf.get_variable(name='kernel3_weight', shape=[5, 50, 64], initializer=tf.random_uniform_initializer(
			minval=-10/(data.max_seq_len + 64), maxval=10/(data.max_seq_len + 64)))

		kernel_bias1 = tf.get_variable(name='kernel1_bias', shape=[64], initializer=cnst_init)
		kernel_bias2 = tf.get_variable(name='kernel2_bias', shape=[64], initializer=cnst_init)
		kernel_bias3 = tf.get_variable(name='kernel3_bias', shape=[64], initializer=cnst_init)

		conv1 = tf.nn.conv1d(embed_words, kernel1, 1, 'VALID') + kernel_bias1
		conv2 = tf.nn.conv1d(embed_words, kernel2, 1, 'VALID') + kernel_bias2
		conv3 = tf.nn.conv1d(embed_words, kernel3, 1, 'VALID') + kernel_bias3

		pool1 = tf.reduce_max(conv1, axis=1)
		pool2 = tf.reduce_max(conv2, axis=1)
		pool3 = tf.reduce_max(conv3, axis=1)

		keep_rate = tf.placeholder(tf.float32, name='keep_rate')
		cnn_output = tf.nn.relu(tf.concat([pool1, pool2, pool3], axis=1))

		true_emojis = tf.placeholder(tf.int32, [data.batch_size], name='emoji_ids')
		true_probs = tf.one_hot(true_emojis, data.emoji_vocab_size)

		cnn_out_dim = 3*64
		hidden_weight = tf.get_variable(name='hidden_weight', shape=[cnn_out_dim, 100],
				initializer=tf.random_uniform_initializer(minval=-1/(cnn_out_dim + 100), maxval=1/(cnn_out_dim + 100)))
		hidden_bias = tf.get_variable(name='hidden_bias', shape=[100], initializer=cnst_init)
		logit_weight = tf.get_variable(name='logit_weight', shape=[100, data.emoji_vocab_size],
				initializer=tf.random_uniform_initializer(minval=-1/(100 + data.emoji_vocab_size),
				maxval=1/(100 + data.emoji_vocab_size)))
		logit_bias = tf.get_variable(name='logit_bias', shape=[data.emoji_vocab_size], initializer=cnst_init)

		hidden = tf.nn.dropout(tf.matmul(cnn_output, hidden_weight) + hidden_bias, keep_prob=keep_rate)
		logits = tf.matmul(hidden, logit_weight) + logit_bias
		prediction = tf.argmax(tf.nn.softmax(logits), axis=1, name='prediction')
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=true_probs, logits=logits), name='loss')

		global_step = tf.get_variable(name='global_step', initializer=tf.constant(0, dtype=tf.int64), trainable=False)
		adam = tf.train.AdadeltaOptimizer()
		gradients = adam.compute_gradients(loss)
		normed_grads = [(tf.clip_by_norm(grad, 5), var) for grad, var in gradients]
		optimizer = adam.apply_gradients(normed_grads, global_step=global_step, name='optimizer')
		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver(max_to_keep=5)

	total_steps = epochs * data.batch_count('train')
	step = sess.run(global_step)

	data.reset_batch('validation')
	total_loss = 0
	batch_count = data.batch_count('validation')
	conf_mat = np.zeros((K, K))
	precision = np.zeros(K)
	recall = np.zeros(K)
	f1 = np.zeros(K)
	for i in xrange(batch_count):
		batch = data.next_batch('validation')

		feed_dict = {
			input_words : batch[0],
			true_emojis : batch[1],
			keep_rate : 1.0
		}
		batch_loss, predicted = sess.run([loss, prediction], feed_dict=feed_dict)
		total_loss += batch_loss

		conf_mat += skl.metrics.confusion_matrix(batch[1], predicted)

	for i in xrange(K):
		precision[i] = conf_mat[i,i] / float(conf_mat[:,i].sum() + 1e-12)
		recall[i] = conf_mat[i,i] / float(conf_mat[i,:].sum() + 1e-12)
		f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i] + 1e-12)

	print("initial validation loss, %2.6f, F1 score %2.6f" % (total_loss / float(batch_count), f1.sum() / K))

	while step < total_steps:
		epoch = step / data.batch_count('train')

		data.reset_batch('train')
		batch_count = data.batch_count('train')
		step = sess.run(global_step) % batch_count
		check = (batch_count / 1000) * 100
		for i in xrange(batch_count):
			batch = data.next_batch('train')
			feed_dict = {
				input_words : batch[0],
				true_emojis : batch[1],
				keep_rate : 0.5
			}
			_, batch_loss, step = sess.run([optimizer, loss, global_step], feed_dict=feed_dict)
			step %= batch_count
			if (step+1) % check == 0:
				print("epoch %d: %d/%d, test loss %2.6f" % (epoch+1, step+1, batch_count, batch_loss))


		data.reset_batch('validation')
		total_loss = 0
		batch_count = data.batch_count('validation')
		conf_mat = np.zeros((K,K), dtype=np.int32)
		for i in xrange(batch_count):
			batch = data.next_batch('validation')

			feed_dict = {
				input_words : batch[0],
				true_emojis : batch[1],
				keep_rate : 1.0
			}
			batch_loss, predicted = sess.run([loss, prediction], feed_dict=feed_dict)
			total_loss += batch_loss

			conf_mat += skl.metrics.confusion_matrix(batch[1], predicted)

		for i in xrange(K):
			precision[i] = conf_mat[i,i] / float(conf_mat[:,i].sum() + 1e-12)
			recall[i] = conf_mat[i,i] / float(conf_mat[i,:].sum() + 1e-12)
			f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i] + 1e-12)

		print("epoch %d: validation loss, %2.6f, F1 score %2.6f" % (epoch+1, total_loss / float(batch_count), f1.sum() / K))
		saver.save(sess, "%s/epoch%d" % (folder, epoch+1))
		step = sess.run(global_step)
		assert((step % data.batch_count('train')) == 0)

	data.reset_batch('test')
	total_loss = 0
	batch_count = data.batch_count('test')
	for i in xrange(batch_count):
		batch = data.next_batch('test')

		feed_dict = {
			input_words : batch[0],
			true_emojis : batch[1],
			keep_rate : 1.0
		}
		total_loss += sess.run(loss, feed_dict=feed_dict)

	print("final test loss, %2.6f" % (total_loss / float(batch_count)))

	# generate performance metrics
	conf_mat = np.zeros((K,K), dtype=np.int32)
	samples = data.batch_count('test')
	data.reset_batch('test')
	for i in xrange(samples):
		batch = data.next_batch('test')
		predicted = sess.run(prediction, feed_dict={
			input_words: batch[0],
			keep_rate: 1.0})
		true_vals = batch[1]

		for j in xrange(100):
			conf_mat[true_vals[j], predicted[j]] += 1

	fig, ax = plt.subplots()
	fig.suptitle('Confusion Matrix')
	plt.xlabel('Predicted Label')
	plt.ylabel('True Label')
	cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)
	for i in xrange(K):
		for j in xrange(K):
			ax.text(i, j, str(conf_mat[j,i]), va='center', ha='center')

	fig.colorbar(cax)
	fig.savefig('base_confusion.png')
	plt.close(fig)

	precision = np.zeros(K)
	recall = np.zeros(K)
	f1 = np.zeros(K)

	for i in xrange(K):
		precision[i] = conf_mat[i,i] / float(conf_mat[:,i].sum() + 1e-12)
		recall[i] = conf_mat[i,i] / float(conf_mat[i,:].sum() + 1e-12)
		f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i] + 1e-12)
	
	print('Precision %s' % precision)
	print('Recall %s' % recall)
	print('F1 Score %s' % f1)
	print('Avg precision %2.6f' % (precision.sum() / K))
	print('Avg recall %2.6f' % (recall.sum() / K))
	print('Avg F1 Score %2.6f' % (f1.sum() / K))

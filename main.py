import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from emoji_cnn import EmojiCNN

K = 5
with tf.Session() as sess:
	model = EmojiCNN(sess,
		data=str(K),
		batch_size=100,
		name='baseline',
		embedding=50,
		kernel_widths=[3,4,57],
		kernel_filters=[64,64,64])
	model.run(epochs=5)
	train_loss, valid_loss = model.train_loss, model.valid_loss

	# generate training data
	fig = plt.figure()
	fig.suptitle('training and validation loss vs epoch')
	plt.xlabel('epoch')
	plt.ylabel('loss')

	train_handle, = plt.plot(train_loss)
	valid_handle, = plt.plot(valid_loss)
	plt.legend([train_handle, valid_handle], ['train', 'validation'])
	fig.savefig("article.png")
	plt.close(fig)

	# generate performance metrics
	conf_mat = np.zeros((K,K), dtype=np.int32)
	samples = model.loader.batch_count('test')
	model.loader.reset_batch('test')
	for i in xrange(samples):
		data = model.loader.next_batch('test')
		predicted = sess.run(model.prediction, feed_dict={model.input_words: data[0], model.keep_rate: 1.0})
		true_vals = data[1]

		for j in xrange(100):
			conf_mat[predicted[j], true_vals[j]] += 1

	fig, ax = plt.subplots()
	ax.matshow(conf_mat, cmap=plt.cm.Blues)
	for i in xrange(K):
		for j in xrange(K):
			ax.text(i, j, str(conf_mat[j,i]), va='center', ha='center')

	fig.savefig('confusion.png')
	plt.close(fig)

	precision = np.zeros(K)
	recall = np.zeros(K)
	f1 = np.zeros(K)

	for i in xrange(K):
		precision[i] = conf_mat[i,i] / float(conf_mat[i,:].sum() + 1e-12)
		recall[i] = conf_mat[i,i] / float(conf_mat[:,i].sum() + 1e-12)
		f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i] + 1e-12)
	
	print('Precision %s' % precision)
	print('Recall %s' % recall)
	print('F1 Score %s' % f1)
	print('Avg F1 Score %2.6f' % (f1.sum() / K))


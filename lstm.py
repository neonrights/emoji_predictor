import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from emoji_char_lstm import EmojiCharLSTM

K = 5
with tf.Session() as sess:
	widths = [1,2,3,4,5,6]
	filters = [25*width for width in widths]
	model = EmojiCharLSTM(sess,
		data=str(K),
		name='basic_lstm',
		kernel_widths=widths,
		kernel_filters=filters,
		lstm_dims=[300],
		highways=1,
		resample=True)
	model.run(epochs=10)
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
	conf_mat = model.confusion_matrix('train')

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


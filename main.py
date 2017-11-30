import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from emoji_cnn import EmojiCNN

K = 20
with tf.Session() as sess:
	model = EmojiCNN(sess,
		data=str(K),
		batch_size=100,
		name='model',
		embed_dim=50,
		kernel_widths=[3, 4, 5],
		kernel_filters=[64, 64 ,64],
		layers=[100],
		weighted=True)
	train_loss, valid_loss = model.run(epochs=5)

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
	samples = model.loader.batches[2]
	model.loader.reset_batch('test')
	for i in xrange(samples):
		data = model.loader.next_batch('test')
		predicted = sess.run(model.prediction, feed_dict={model.input_words: data[0], model.keep_rate: 1.0})
		true_vals = np.argmax(data[1], axis=1)

		for j in xrange(100):
			conf_mat[true_vals[j], predicted[j]] += 1

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
		precision[i] = conf_mat[i,i] / float(conf_mat[:,i].sum() + 1e-12)
		recall[i] = conf_mat[i,i] / float(conf_mat[i,:].sum() + 1e-12)
		f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
	
	print('Precision %s' % precision)
	print('Recall %s' % recall)
	print('F1 Score %s' % f1)
	print('Avg F1 Score %2.6f' % (f1.sum() / K))


import tensorflow as tf
import os

from loader import *

class EmojiCNN:
	def __init__(self, sess, data, batch_size, name, embed_dim, kernels, layers):
		# initialize batch loader
		self.loader = WordLoader(data=data, batch_size=batch_size)
		self.sess = sess
		self.name = name

		with tf.variable_scope(self.name):
			# embed words
			self.input_words = tf.placeholder(tf.int64, [None, None], name='word-ids')
			word_embeds = tf.get_variable("word-embedding", [self.loader.word_vocab_size, embed_dim])
			embed_words = tf.nn.embedding_lookup(word_embeds, self.input_words)

			# initializers for weights and biases
			trnc_norm_init = tf.truncated_normal_initializer(stddev=0.1)
			cnst_init = tf.constant_initializer(0.1)

			# create convolutions
			with tf.variable_scope('conv'):
				outputs = list()
				for width, output in kernels:
					kernel = tf.get_variable(name="kernel-%s-%s" % (width, output),
							shape=[width, embed_dim, output], initializer=trnc_norm_init)
					bias = tf.get_variable(name="kernel-bias-%s-%s" % (width, output),
							shape=[output],	initializer=cnst_init)
					conv = tf.nn.conv1d(embed_words, kernel, 1, 'VALID') + bias
					pool = tf.reduce_max(conv, axis=1)
					outputs.append(pool)

				self.cnn_output = tf.nn.relu(tf.concat(outputs, axis=1))

			# create fully connected layer
			with tf.variable_scope('full'):
				hidden = self.cnn_output
				self.keep_rate = tf.placeholder(tf.float32, name='keep-rate')
				for i, dim in enumerate(layers):
					weights = tf.get_variable(name="hidden-%i-weight" % (i+1),
							shape=[hidden.get_shape()[1], dim],	initializer=trnc_norm_init)
					bias = tf.get_variable(name="hidden-%i-bias" % (i+1),
							shape=[dim], initializer=cnst_init)
					hidden = tf.nn.relu(tf.matmul(hidden, weights) + bias)
					hidden = tf.nn.dropout(hidden, keep_prob=self.keep_rate, name="hidden-%s-output" % (i+1))

				self.full_output = hidden

			# output emoji softmax prediction
			self.true_emojis = tf.placeholder(tf.int64, [None, self.loader.emoji_vocab_size], name='emoji-ids')
			weight = tf.get_variable(name="softmax-weight",
					shape=[self.full_output.get_shape()[1], self.loader.emoji_vocab_size],
					initializer=trnc_norm_init)
			bias = tf.get_variable(name="softmax-bias",	shape=[self.loader.emoji_vocab_size],
					initializer=cnst_init)
			self.output = tf.nn.relu(tf.matmul(self.full_output, weight) + bias)
			self.prediction = tf.nn.softmax(self.output)
			self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
					labels=tf.squeeze(self.true_emojis), logits=self.output))
			# create f1 score

			self.trainer = tf.train.AdamOptimizer().minimize(self.cross_entropy)
			self.sess.run(tf.global_variables_initializer())
			self.saver = tf.train.Saver()

	# train model
	def train(self, epoch):
		self.loader.reset_batch(dataset='train')
		for i in xrange(self.loader.batches[0]):
			data = self.loader.next_batch(dataset='train')

			if data is None:
				print("training prematurely stopped")
				break

			feed_dict = {
				self.input_words : data[0],
				self.true_emojis : data[1],
				self.keep_rate : 0.5
			}
			_, loss = self.sess.run([self.trainer, self.cross_entropy], feed_dict=feed_dict)
			if (i+1) % 50 == 0:
				print("epoch %d: %d/%d, test loss %2.6f" % (epoch, i+1, self.loader.batches[0], loss))

	# test model on specified dataset
	def test(self, dataset):
		self.loader.reset_batch(dataset=dataset)
		total_loss
		for i in xrange(self.loader.batches[self.DATA2ID[dataset]]):
			data = self.loader.next_batch(dataset=dataset)

			if data is None:
				print("testing prematurely stopped")
				break

			feed_dict = {
				self.input_words : data[0],
				self.true_emojis : data[1],
				self.keep_rate : 1.0
			}
			total_loss += self.sess.run([loss], feed_dict=feed_dict)

		return float(total_loss) / self.loader.batches[self.DATA2ID[dataset]]

	def predict(self, sentence):
		input_tensor = self.loader.sentence2tensor(sentence)
		feed_dict = {
			self.input_words : input_tensor,
			self.keep_rate : 1.0
		}
		predicted = self.sess.run([self.prediction], feed_dict=feed_dict)
		return self.loader.id2emoji[predicted]


	def save(self, key):
		if not os.path.isdir('tmp'):
			os.mkdir('tmp')

		self.saver.save(sess, "tmp/%s-%s.ckpt" % (self.name, key))


	# train, save, and test model
	def run(self, epochs=100):
		for e in xrange(epochs):
			self.train(e)
			loss = self.test(dataset='validation')
			print("epoch %d: validation loss %2.6f" % (e+1, loss))
			self.save(key='epoch%d' % (e+1))
		
		self.save(key='final')
		self.test(dataset='train')
		self.test(dataset='validation')
		self.test(dataset='test')


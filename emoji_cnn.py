import tensorflow as tf
import sklearn as sk
import os
import pickle
import glob

from loader import *

class EmojiCNN:
	def __init__(self, sess, data, batch_size, name, embed_dim, kernel_widths, kernel_filters, layers, restore=False):
		self.sess = sess
		self.name = name
		self.loader = WordLoader(data=data, batch_size=batch_size)

		try: # to restore object if desired
			if not restore:
				raise IOError

			print("attempting to restore model")

			# get most recent file
			files = glob.glob("tmp_%s/*.meta" % self.name)
			if not files:
				raise IOError

			files.sort(key=lambda x: -os.path.getmtime(x))
			meta_file = files[0]
			checkpoint_name = meta_file.split('.')[0]

			self.saver = tf.train.import_meta_graph(meta_file)
			self.saver.restore(self.sess, checkpoint_name)

			# restore class variables
			if os.path.isfile("tmp_%s/loss.pkl" % self.name):
				with open("tmp_%s/loss.pkl" % self.name, 'r') as f:
					self.train_loss, self.valid_loss = pickle.load(f)
			else:
				self.train_loss = list()
				self.valid_loss = list()

			graph = tf.get_default_graph()
			self.input_words = graph.get_tensor_by_name('%s/word_ids:0' % self.name)
			self.true_emojis = graph.get_tensor_by_name('%s/emoji_ids:0' % self.name)
			self.keep_rate = graph.get_tensor_by_name('%s/full/keep_rate:0' % self.name)
			self.output = graph.get_tensor_by_name('%s/output:0' % self.name)
			self.prediction = graph.get_tensor_by_name('%s/prediction:0' % self.name)
			self.loss = graph.get_tensor_by_name('%s/loss:0' % self.name)
			self.global_step = graph.get_tensor_by_name('%s/global_step:0' % self.name)
			self.trainer = graph.get_operation_by_name('%s/trainer' % self.name)

			print("restored from %s" % checkpoint_name)

		except (IOError, tf.errors.NotFoundError) as e: # initialize object as normal
			if restore: # if failed to restore, reset session
				print("failed to restore model")
				#tf.reset_default_graph() clear graph

			print("building model")
			with tf.variable_scope(self.name):
				# embed words
				self.input_words = tf.placeholder(tf.int64, [None, None], name='word_ids')
				word_embeds = tf.get_variable("word_embedding", [self.loader.word_vocab_size, embed_dim])
				embed_words = tf.nn.embedding_lookup(word_embeds, self.input_words)

				# initializers for weights and biases
				trnc_norm_init = tf.truncated_normal_initializer(stddev=0.1)
				cnst_init = tf.constant_initializer(0.1)

				# create convolutions
				with tf.variable_scope('conv'):
					outputs = list()
					for width, filters in zip(kernel_widths, kernel_filters):
						kernel = tf.get_variable(name="kernel_%s_%s" % (width, filters),
								shape=[width, embed_dim, filters], initializer=trnc_norm_init)
						bias = tf.get_variable(name="kernel_bias_%s_%s" % (width, filters),
								shape=[filters],	initializer=cnst_init)
						conv = tf.nn.conv1d(embed_words, kernel, 1, 'VALID') + bias
						pool = tf.reduce_max(conv, axis=1)
						outputs.append(pool)

					cnn_output = tf.nn.relu(tf.concat(outputs, axis=1))

				# create fully connected layer
				with tf.variable_scope('full'):
					hidden = cnn_output
					self.keep_rate = tf.placeholder(tf.float32, name='keep_rate')
					for i, dim in enumerate(layers):
						weights = tf.get_variable(name="hidden_%i_weight" % (i+1),
								shape=[hidden.get_shape()[1], dim],	initializer=trnc_norm_init)
						bias = tf.get_variable(name="hidden_%i_bias" % (i+1),
								shape=[dim], initializer=cnst_init)
						hidden = tf.nn.relu(tf.matmul(hidden, weights) + bias)
						hidden = tf.nn.dropout(hidden, keep_prob=self.keep_rate, name="hidden_%s_output" % (i+1))

					full_output = hidden

				# output emoji softmax prediction
				self.true_emojis = tf.placeholder(tf.int64, [None, self.loader.emoji_vocab_size], name='emoji_ids')
				weight = tf.get_variable(name="softmax_weight",
						shape=[full_output.get_shape()[1], self.loader.emoji_vocab_size],
						initializer=trnc_norm_init)
				bias = tf.get_variable(name="softmax_bias",	shape=[self.loader.emoji_vocab_size],
						initializer=cnst_init)
				self.output = tf.nn.relu(tf.matmul(full_output, weight) + bias, name='output')
				self.prediction = tf.nn.softmax(self.output, name='prediction')
				
				# metrics
				self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
						labels=tf.squeeze(self.true_emojis), logits=self.output), name='loss')

				# optimizer and tracking
				self.global_step = tf.get_variable(name='global_step',
						initializer=tf.constant(0, dtype=tf.int64), trainable=False)
				self.trainer = tf.train.AdamOptimizer().minimize(self.loss,
						global_step=self.global_step, name='trainer')
				
				# ready to go
				self.sess.run(tf.global_variables_initializer())
				self.saver = tf.train.Saver()
				self.train_loss = list()
				self.valid_loss = list()

	# train model
	def train(self, epoch):
		self.loader.reset_batch(dataset='train')
		batch_count = self.loader.batches[0]
		step = self.sess.run(self.global_step) % batch_count
		check = (batch_count / 1000) * 100
		for i in xrange(batch_count):
			data = self.loader.next_batch(dataset='train')

			if data is None:
				print("training prematurely stopped")
				break

			feed_dict = {
				self.input_words : data[0],
				self.true_emojis : data[1],
				self.keep_rate : 0.5
			}
			_, loss, step = self.sess.run([self.trainer, self.loss, self.global_step], feed_dict=feed_dict)
			step %= batch_count
			if (step+1) % check == 0:
				print("epoch %d: %d/%d, test loss %2.6f" % (epoch, step+1, self.loader.batches[0], loss))

			step = self.sess.run(self.global_step) % batch_count

	# test model on specified dataset
	def test(self, dataset):
		self.loader.reset_batch(dataset=dataset)
		total_loss = 0
		batch_count = self.loader.batches[self.loader.DATA2ID[dataset]]
		for i in xrange(batch_count):
			data = self.loader.next_batch(dataset=dataset)

			if data is None:
				print("testing prematurely stopped")
				break

			feed_dict = {
				self.input_words : data[0],
				self.true_emojis : data[1],
				self.keep_rate : 1.0
			}
			batch_loss, emoji_pred, emoji_true = self.sess.run([self.loss, self.prediction, self.true_emojis], feed_dict=feed_dict)


		return float(total_loss) / batch_count


	def predict(self, sentence):
		input_tensor = self.loader.sentence2tensor(sentence)
		feed_dict = {
			self.input_words : input_tensor,
			self.keep_rate : 1.0
		}
		predicted = self.sess.run(self.prediction, feed_dict=feed_dict)
		return self.loader.id2emoji[predicted]


	def save(self, key):
		if not os.path.isdir('tmp_%s' % self.name):
			os.mkdir('tmp_%s' % self.name)
		
		# pickle picklable self objects
		if self.train_loss and self.valid_loss:
			with open('tmp_%s/loss.pkl' % self.name, 'w+') as f:
				pickle.dump((self.train_loss, self.valid_loss), f)

		self.saver.save(self.sess, "tmp_%s/%s" % (self.name, key))
		print("saved %s" % key)

	# train, save, and test model
	def run(self, epochs=100):
		total_steps = epochs * self.loader.batches[0]
		step = self.sess.run(self.global_step)

		if step == 0:
			self.save('initial')
			self.train_loss.append(self.test(dataset='train'))
			self.valid_loss.append(self.test(dataset='validation'))
		
		print("initial train and validation loss: %2.6f %2.6f" % (self.train_loss[0], self.valid_loss[0]))
		while step < total_steps:
			epoch = step / self.loader.batches[0]
			self.train(epoch+1)

			self.train_loss.append(self.test(dataset='train'))
			self.valid_loss.append(self.test(dataset='validation'))

			print("epoch %d: train and validation loss, %2.6f %2.6f" % (epoch+1, self.train_loss[-1], self.valid_loss[-1]))
			self.save('epoch%d' % (epoch+1))

			step = self.sess.run(self.global_step)
			assert((step % self.loader.batches[0]) == 0)
		
		self.save('final')
		test_loss = self.test(dataset='test')
		print("final test loss %2.6f" % test_loss)
		return self.train_loss, self.valid_loss


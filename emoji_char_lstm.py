import tensorflow as tf
import numpy as np
import os
import pickle
import glob

from loader import CharLoader

class EmojiCharLSTM:
		def __init__(self, sess, data, name, kernel_widths, kernel_filters,
				batch_size=100,	embedding=15, learn_rate=1.0, lstm_dims=[100],
				highways=0, max_gradient_norm=5.0, resample=False, restore=None):
		self.sess = sess
		self.name = name
		self.loader = CharLoader(data=data, batch_size=batch_size, resample=resample)

		try: # to restore object if desired
			if not restore:
				raise IOError

			if os.path.isfile("tmp_%s/%s.meta" % (self.name, restore)):
				print("attempting to restore model from %s" % restore)
				meta_file = "tmp_%s/%s.meta" % (self.name, restore)
			else:
				# get most recent file
				print("attempting to restore model from latest checkpoint")
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
					self.train_loss, self.valid_loss, self.rate = pickle.load(f)
			else:
				self.train_loss = list()
				self.valid_loss = list()

			graph = tf.get_default_graph()
			self.input_chars = graph.get_tensor_by_name('%s/char_ids:0' % self.name)
			self.true_emojis = graph.get_tensor_by_name('%s/emoji_ids:0' % self.name)
			self.keep_rate = graph.get_tensor_by_name('%s/full/keep_rate:0' % self.name)
			self.prediction = graph.get_tensor_by_name('%s/prediction:0' % self.name)
			self.loss = graph.get_tensor_by_name('%s/loss:0' % self.name)
			self.learn_rate = graph.get_tensor_by_name('%s/learn_rate:0' % self.name)
			self.global_step = graph.get_tensor_by_name('%s/global_step:0' % self.name)
			self.trainer = graph.get_operation_by_name('%s/trainer' % self.name)

			print("restored from %s" % checkpoint_name)

		except (IOError, tf.errors.NotFoundError) as e: # initialize object as normal
			if restore: # if failed to restore, reset session
				print("failed to restore model")
				#tf.reset_default_graph() clear graph

			print("building model")
			with tf.variable_scope(self.name):
				# embed chars
				self.input_chars = tf.placeholder(tf.int32,
						[None, self.loader.max_seq_len, self.loader.max_word_len], name='char_ids')
				char_embeds = tf.get_variable("char_embedding", shape=[self.loader.char_vocab_size, embedding]
						initializer=tf.random_uniform_initializer(minval=-0.5, maxval=0.5))

				# initializers for weights and biases
				unif_init = tf.random_uniform_initializer(minval=-0.05, maxval=0.05)
				cnst_init = tf.constant_initializer(0.1)

				# create convolutions
				with tf.variable_scope('conv'):
					cnn_outputs = list()
					char_indices = tf.split(self.input_chars, self.loader.max_seq_len, 1)
					for i in xrange(self.loader.max_seq_len):
						embedded_chars = tf.nn.embedding_lookup(char_embeds, char_indices[i])

						temp_output = list()
						for width, filters in zip(kernel_widths, kernel_filters):
							kernel = tf.get_variable(name="kernel_%s_%s" % (width, filters),
									shape=[width, embedding, filters], initializer=unif_init)
							bias = tf.get_variable(name="kernel_bias_%s_%s" % (width, filters),
									shape=[filters], initializer=cnst_init)
							conv = tf.nn.conv1d(embedded_chars, kernel, 1, 'VALID') + bias
							pool = tf.reduce_max(conv, axis=1)
							temp_output.append(pool)

						cnn_outputs.append(tf.concat(temp_output, axis=1))

				# initializer and expected cnn output dimension
				N = sum([width*filters for width, filters in zip(kernel_widths, kernel_filters)])
				neg_init = tf.constant_initializer(-1)

				# create highway network
				with tf.variables_scope('hwy'):
					hwy_inputs = cnn_outputs
					if highways > 0:
						for i in xrange(highways):
							hwy_outputs = list()
							W_T = tf.get_variable(name="transform_%d_weight" % (i+1),
									shape=[N, N], initializer=unif_init)
							b_T = tf.get_variable(name="transform_%d_bias" % (i+1),
									shape=[N], initializer=neg_init)
							W_H = tf.get_variable(name="carry_%d_weight" % (i+1),
									shape=[N, N], initializer=unif_init)
							b_H = tf.get_variable(name="carry_%d_bias" % (i+1),
									shape=[N], initializer=neg_init)
							for hwy_input in hwy_inputs:
								trans_gate = tf.sigmoid(tf.matmul(hwy_input, W_T) + b_T)
							    trans_output = trans_gate * (tf.nn.relu(tf.matmul(hwy_input, W_H)) + b_H)
								carry_output = (1 - trans_gate) * hwy_input
								hwy_outputs.append(trans_output + carry_output)

							hwy_inputs = hwy_outputs
					else:
						hwy_outputs = hwy_inputs

				# create lstms
				with tf.variable_scope('lstm'):
					lstm_inputs = hwy_outputs
					self.keep_rate = tf.placeholder(tf.float32, name='keep_rate')
					lstm_cells = list()
					for i, dim in enumerate(lstm_dims):
						if i == 0:
							lstm_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCELL(dim),
									name="lstm_cell_%d" % (i+1))
						else:
							lstm_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCELL(dim),
									input_keep_prob=self.keep_rate, name="lstm_cell_%d" % (i+1))
						
						lstm_cells.append(lstm_cell)

					stacked_lstms = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)
					init_state = stacked_lstms.zero_state(self.input_chars.get_shape()[0])
					lstm_outputs, _ = tf.contrib.rnn.static_rnn(stacked_lstms, lstm_inputs, initial_state=init_state)
					lstm_final_output = lstm_outputs[-1]

				# output emoji softmax prediction
				self.true_emojis = tf.placeholder(tf.int32, [None], name='emoji_ids')
				true_probs = tf.one_hot(self.true_emojis, self.loader.emoji_vocab_size)

				weight = tf.get_variable(name="softmax_weight",
						shape=[lstm_dims[-1], self.loader.emoji_vocab_size],
						initializer=trnc_norm_init)
				bias = tf.get_variable(name="softmax_bias",	shape=[self.loader.emoji_vocab_size],
						initializer=cnst_init)

				output = tf.matmul(tf.nn.dropout(lstm_final_output, keep_prob=self.keep_rate), weight) + bias
				self.prediction = tf.argmax(tf.nn.softmax(output), axis=1, name='prediction')
				self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
						labels=true_probs, logits=output), name='loss')

				# optimizer and tracking
				trainables = tf.trainable_variables()
				gradients = list()
				for grad in tf.gradients(self.loss, trainables):
					if grad is not None:
						gradients.append(tf.clip_by_nrom(grad, gradient_norm))
					else:
						gradients.append(grad)

				self.rate = learn_rate
				self.learn_rate = tf.placeholder(tf.float32, name='learn_rate', trainable=False)
				self.global_step = tf.Variable(0, name='global_step', trainable=False)
				optimizer = tf.train.GradientDescentOptimizer(self.learn_rate)
				self.trainer = optimizer.apply_gradients(zip(gradients, trainables),
						global_step=self.global_step, name='trainer')
				
				# ready to go
				self.sess.run(tf.global_variables_initializer())
				self.saver = tf.train.Saver()
				self.train_loss = list()
				self.valid_loss = list()
				print("model built")

	# train model
	def train(self, epoch):
		self.loader.reset_batch(dataset='train')
		batch_count = self.loader.batch_count('train')
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
				self.keep_rate : 0.5,
				self.learn_rate : self.rate
			}
			_, loss, step = self.sess.run([self.trainer, self.loss,
					self.global_step], feed_dict=feed_dict)
			step %= batch_count
			if (step+1) % check == 0:
				print("epoch %d: %d/%d, test loss %2.6f" % (epoch, step+1,
						self.loader.batch_count('train'), loss))

			step = self.sess.run(self.global_step) % batch_count

	# test model on specified dataset
	def test(self, dataset):
		self.loader.reset_batch(dataset)
		total_loss = 0
		for i in xrange(self.loader.batch_count(dataset)):
			data = self.loader.next_batch(dataset)

			if data is None:
				print("testing prematurely stopped")
				break

			feed_dict = {
				self.input_words : data[0],
				self.true_emojis : data[1],
				self.keep_rate : 1.0
			}
			total_loss += self.sess.run(self.loss, feed_dict=feed_dict)

		return float(total_loss) / self.loader.batch_count(dataset)


	def predict(self, sentence):
		if type(sentence) is str:
			sentence = self.loader.sentence2tensor(sentence)

		feed_dict = {
			self.input_words : sentence,
			self.keep_rate : 1.0
		}
		predicted = self.sess.run(self.prediction, feed_dict=feed_dict)
		return [self.loader.id2emoji(predict+1) for predict in predicted]


	def save(self, key):
		if not os.path.isdir('tmp_%s' % self.name):
			os.mkdir('tmp_%s' % self.name)
		
		# pickle picklable self objects
		if self.train_loss and self.valid_loss:
			with open('tmp_%s/loss.pkl' % self.name, 'w+') as f:
				pickle.dump((self.train_loss, self.valid_loss, self.rate), f)

		self.saver.save(self.sess, "tmp_%s/%s" % (self.name, key))
		print("saved %s" % key)

	# train, save, and test model
	def run(self, epochs=100):
		total_steps = epochs * self.loader.batch_count('train')
		step = self.sess.run(self.global_step)

		if step == 0:
			self.train_loss.append(self.test(dataset='train'))
			self.valid_loss.append(self.test(dataset='validation'))
		
		print("initial train and validation loss: %2.6f %2.6f" %
				(self.train_loss[0], self.valid_loss[0]))

		while step < total_steps:
			epoch = step / self.loader.batch_count('train')
			self.train(epoch+1)

			self.train_loss.append(self.test(dataset='train'))
			self.valid_loss.append(self.test(dataset='validation'))

			if self.train_loss[-1] > (train_loss[-2] - 1):
				self.rate /= 2

			print("epoch %d: train and validation loss, %2.6f %2.6f" % (epoch+1,
					self.train_loss[-1], self.valid_loss[-1]))
			self.save('epoch%d' % (epoch+1))

			step = self.sess.run(self.global_step)
			assert((step % self.loader.batch_count('train')) == 0)
		
		self.save('final')
		test_loss = self.test(dataset='test')
		print("final test loss %2.6f" % test_loss)
		return self.train_loss, self.valid_loss
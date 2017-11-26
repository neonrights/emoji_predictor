import numpy as np
import pandas as pd
import os
import pickle

from collections import Counter

PATH = './data/'


def save(obj, filename):
	with open(filename, 'w+') as f:
		pickle.save(obj, f)


def load(filename):
	with open(filename, 'r') as f:
		return pickle.load(f)


class WordLoader:
	def __init__(self, data, batchsize=100):
		self.batchsize = batchsize
		try:
			self.word2id, self.id2word = load('%s_word_vocab.pkl' % data)
			self.word_tensors = load('%s_word_tensor.pkl' % data)
			self.max_seq_len = self.word_tensors[0].shape[1]
			self.samples = [tensor.shape[0] for tensor in self.word_tensors]
		except IOError:
			self._build_vocab(data)

		self.vocab_size = len(id2word)
		self.batch_ptr = [0, 0, 0]
		self.batch_count = [0, 0, 0]
		self.batch_tensors = [0, 0, 0]
		for idx in xrange(3):
			offset = self.samples[idx] % self.batchsize
			self.batch_count[idx] = self.samples[idx] / self.batchsize
			batch_tensor = self.word_tensors[:-offset,:]
			batch_tensor = np.vsplit(batch_tensor, self.batch_count[idx])
			self.batch_tensors[idx] = batch_tensor


	def _build_vocab(self, data):
		# create vocab and ids from either character or words
		filenames = ['%s_train' % data, '%s_valid' % data, '%s_test' % data]
		self.word2id = {' ': 0}
		self.id2word = [' ']
		self.max_seq_len = 0
		self.samples = list()
		counts = Counter()
		for filename in filenames:
			with open(filename, 'r') as f:
				sample = 0
				for line in f:
					sample += 1
					words = line.split()
					self.max_seq_len = max(self.max_seq_len, len(words))
					for word in words:
						# add targets
						counts[word] += 1
			self.samples.append(sample)

		for word in counts.most_common():
			self.word2id[word] = len(id2word)
			self.id2word.append(word)

		save((self.word2id, self.id2word), '%s_word_vocab.pkl' % data)

		# create tensor from word ids
		self.word_tensors = dict()
		for idx, filename in enumerate(filenames):
			word_tensor = np.zeros(samples[idx], self.max_seq_len)
			with open(filename, 'r') as f:
				for i, line in enumerate(f):
					for j, word in enumerate(line.split()):
						word_tensor[i, j] = self.word2id[word]
			self.word_tensors.append(word_tensor)

		save(self.word_tensors, '%s_word_tensor.pkl' % data)


	def next_batch(self, set='train'):
		pass


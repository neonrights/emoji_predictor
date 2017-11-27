import numpy as np
import pandas as pd
import pickle
import string

from collections import Counter
from os import path


def save(obj, filename):
	with open(filename, 'w+') as f:
		pickle.dump(obj, f)

def load(filename):
	with open(filename, 'r') as f:
		return pickle.load(f)

DATA2ID = {'train': 0, 'validation': 1, 'test': 2}
PATH = './data/'

class WordLoader:
	def __init__(self, data='5', batch_size=100):
		self.batch_size = batch_size
		try: # to load saved vocab data
			print('loading saved vocabulary and processed data')
			self.word2id, self.id2word, self.emoji2id, self.id2emoji = load('%s_word_vocab.pkl' % data)
			self.word_tensors, self.emoji_tensors = load('%s_word_tensor.pkl' % data)
			self.max_seq_len = self.word_tensors[0].shape[1]
			self.samples = [tensor.shape[0] for tensor in self.word_tensors]
		except IOError:
			print('failed to load, building vocabulary and processed data')
			self._build_vocab(data)

		self.word_vocab_size = len(self.id2word)
		self.emoji_vocab_size = len(self.emoji2id)

		# reshape data into batches
		self.batch_num = [0, 0, 0]
		self.batches = list()
		temp_word_tensors = list()
		temp_emoji_tensors = list()
		for idx in xrange(3):
			# remove extra data samples
			offset = self.samples[idx] % self.batch_size
			self.batches.append(self.samples[idx] / self.batch_size)

			word_tensor = self.word_tensors[idx]
			emoji_tensor = self.emoji_tensors[idx]
			if offset != 0:
				word_tensor = word_tensor[:-offset,:]
				emoji_tensor = emoji_tensor[:-offset]

			word_tensor = np.vsplit(word_tensor, self.batches[idx])
			emoji_tensor = np.split(emoji_tensor, self.batches[idx])

			temp_word_tensors.append(word_tensor)
			temp_emoji_tensors.append(emoji_tensor)

		self.word_tensors = temp_word_tensors
		self.emoji_tensors = temp_emoji_tensors

	def _build_vocab(self, data):
		# create vocab and ids from either character or words
		filenames = [
			path.join(PATH, '%s_train' % data),
			path.join(PATH, '%s_validation' % data),
			path.join(PATH, '%s_test' % data)
		]

		# get counts and samples
		self.max_seq_len = 0
		self.samples = list()
		word_counts = Counter()
		emoji_counts = Counter()
		for filename in filenames:
			with open(filename, 'r') as f:
				line_count = 0
				for line in f:
					line = line.translate(None, string.punctuation) # strip punctuation
					words = line.split()
					emoji_counts[words[-1]] += 1
					words = words[:-1]
					line_count += 1
					self.max_seq_len = max(self.max_seq_len, len(words))
					for word in words:
						word_counts[word] += 1

			self.samples.append(line_count)

		# build word vocab
		self.word2id = {' ': 0}
		self.id2word = [' ']
		for word, _ in word_counts.most_common():
			self.word2id[word] = len(self.id2word)
			self.id2word.append(word)

		# build emoji vocab
		self.emoji2id = {' ': 0}
		self.id2emoji = [' ']
		for emoji, _ in emoji_counts.most_common():
			self.emoji2id[emoji] = len(self.id2emoji)
			self.id2emoji.append(emoji)

		# create tensor from word ids
		self.word_tensors = list()
		self.emoji_tensors = list()
		for idx, filename in enumerate(filenames):
			word_tensor = np.zeros((self.samples[idx], self.max_seq_len), dtype=np.int64)
			emoji_tensor = np.zeros(self.samples[idx], dtype=np.int64)
			with open(filename, 'r') as f:
				for i, line in enumerate(f):
					line = line.translate(None, string.punctuation)
					words = line.split()
					emoji_tensor[i] = self.emoji2id[words[-1]]
					words = words[:-1]
					for j, word in enumerate(words):
						word_tensor[i, j] = self.word2id[word]

			self.word_tensors.append(word_tensor)
			self.emoji_tensors.append(emoji_tensor)

		# saved vocabulary and processed data
		save([self.word2id, self.id2word, self.emoji2id, self.id2emoji], '%s_word_vocab.pkl' % data)
		save([self.word_tensors, self.emoji_tensors], '%s_word_tensor.pkl' % data)


	def next_batch(self, dataset='train'):
		idx = DATA2ID[dataset]
		if self.batch_num[idx] >= self.batches[idx]: # reset, new epoch
			self.batch_num[idx] = 0
			return None
		else:
			return self.word_tensors[idx][self.batch_num[idx]]


	def batch_number(self, dataset='train'):
		return self.batch_num[DATA2ID[dataset]]


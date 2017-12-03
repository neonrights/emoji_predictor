import numpy as np
import pandas as pd
import pickle
import string
import os

from collections import Counter
from pandas import read_csv


def save(obj, filename):
	with open(filename, 'w+') as f:
		pickle.dump(obj, f)

def load(filename):
	with open(filename, 'r') as f:
		return pickle.load(f)


class WordLoader:
	DATA2ID = {'train': 0, 'validation': 1, 'test': 2}
	PATH = './data'

	def __init__(self, data, batch_size=100, glove=25, resample=False):
		self.batch_size = batch_size
		try: # to load saved vocab data
			print('loading saved vocabulary and processed data')
			self.word2id, self.id2word, self.emoji2id, self.id2emoji, self.weights = load(os.path.join(self.PATH, '%s_word_vocab.pkl' % data))
			self.word_tensors, self.emoji_tensors = load(os.path.join(self.PATH, '%s_word_tensor.pkl' % data))
			self.glove_embed = load(os.path.join(self.PATH, '%s_%d_glove.pkl' % (data, glove)))
			self.max_seq_len = self.word_tensors[0].shape[1]
			self.samples = [tensor.shape[0] for tensor in self.word_tensors]
			self.word_vocab_size = len(self.id2word)
			self.emoji_vocab_size = len(self.emoji2id)
		except (IOError, EOFError) as e:
			print('failed to load, building vocabulary and processed data')
			self._build_vocab(data, glove)

		print("%d words, %d emojis" % (self.word_vocab_size-2, self.emoji_vocab_size))

		if resample:
			self._resample(data)
			self.samples[0] = self.emoji_tensors[0].shape[0]

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
				word_tensor = word_tensor[:-offset, :]
				emoji_tensor = emoji_tensor[:-offset]

			word_tensor = np.vsplit(word_tensor, self.batches[idx])
			emoji_tensor = np.split(emoji_tensor, self.batches[idx])

			temp_word_tensors.append(word_tensor)
			temp_emoji_tensors.append(emoji_tensor)

		self.word_tensors = temp_word_tensors
		self.emoji_tensors = temp_emoji_tensors

	# resample training data to even out classes
	def _resample(self, data):
		try:
			print("attempting to load resampled data")
			self.word_tensors[0], self.emoji_tensors[0] = load(os.path.join(self.PATH, '%s_resample.pkl' % data))
		except (IOError, EOFError) as e:
			print("failed to load, building resampled data")
			n_resamples = self.samples[0] / self.emoji_vocab_size
			temp_emoji_tensors = list()
			temp_word_tensors = list()
			for i in xrange(self.emoji_vocab_size):
				indices = np.where(self.emoji_tensors[0] == i)[0]
				r_indices = np.random.choice(indices, size=n_resamples, replace=True)
				temp_word_tensors.append(self.word_tensors[0][r_indices, :])
				temp_emoji_tensors.append(self.emoji_tensors[0][r_indices])

			shuffled = np.arange(n_resamples * self.emoji_vocab_size)
			np.random.shuffle(shuffled)
			self.word_tensors[0] = np.concatenate(temp_word_tensors, axis=0)[shuffled, :]
			self.emoji_tensors[0] = np.concatenate(temp_emoji_tensors)[shuffled]

			save([self.word_tensors[0], self.emoji_tensors[0]], os.path.join(self.PATH, '%s_resample.pkl' % data))


	def _build_vocab(self, data, glove):
		# create vocab and ids from either character or words
		filenames = [
			os.path.join(self.PATH, '%s_train' % data),
			os.path.join(self.PATH, '%s_validation' % data),
			os.path.join(self.PATH, '%s_test' % data)
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
					words = line.split()
					emoji_counts[words[-1]] += 1
					words = words[:-1]
					line_count += 1
					self.max_seq_len = max(self.max_seq_len, len(words))
					for word in words:
						word_counts[word] += 1

			self.samples.append(line_count)

		# build word vocab
		self.word2id = {'<none>': 0, '<unk>': 1}
		self.id2word = ['<none>', '<unk>']
		for word, _ in word_counts.most_common():
			self.word2id[word] = len(self.id2word)
			self.id2word.append(word)
			
		# build emoji vocab
		self.emoji2id = dict()
		self.id2emoji = list()
		self.weights = np.zeros(len(emoji_counts))
		self.emoji_vocab_size = len(emoji_counts)
		total_samples = sum(self.samples)
		for emoji, count in emoji_counts.most_common():
			self.emoji2id[emoji] = len(self.id2emoji)
			self.id2emoji.append(emoji)
			self.weights[self.emoji2id[emoji]] = total_samples / count

		self.word_vocab_size = len(self.id2word)
		self.emoji_vocab_size = len(self.id2emoji)

		self.glove_embed = np.zeros((self.word_vocab_size, glove))
		vocab = set(self.word2id.keys())
		with open('GloVe/glove.twitter.27B.%dd.txt' % glove, 'r') as f:
			for line in f:
				tokens = line.split()
				if tokens[0] in vocab:
					vocab.remove(tokens[0])
					self.glove_embed[self.word2id[tokens[0]], :] = np.array(tokens[1:], dtype=np.float32)

		print("lost %d out of %d" % (len(vocab), self.word_vocab_size))
		for word in vocab:
			self.glove_embed[self.word2id[word], :] = np.random.uniform(low=-1.0, high=1.0, size=glove)

		# create tensor from word ids
		self.word_tensors = list()
		self.emoji_tensors = list()
		for idx, filename in enumerate(filenames):
			word_tensor = np.zeros((self.samples[idx], self.max_seq_len), dtype=np.int64)
			emoji_tensor = np.zeros((self.samples[idx]), dtype=np.int64)
			with open(filename, 'r') as f:
				for i, line in enumerate(f):
					words = line.split()
					emoji_tensor[i] = self.emoji2id[words[-1]]
					words = words[:-1]
					for j, word in enumerate(words):
						word_tensor[i, j] = self.word2id[word]

			self.word_tensors.append(word_tensor)
			self.emoji_tensors.append(emoji_tensor)

		# saved vocabulary and processed data
		save([self.word2id, self.id2word, self.emoji2id, self.id2emoji, self.weights], os.path.join(self.PATH, '%s_word_vocab.pkl' % data))
		save([self.word_tensors, self.emoji_tensors], os.path.join(self.PATH, '%s_word_tensor.pkl' % data))
		save(self.glove_embed, os.path.join(self.PATH, '%s_%d_glove.pkl' % (data, glove)))


	def next_batch(self, dataset='train'):
		idx = self.DATA2ID[dataset]
		if self.batch_num[idx] >= self.batches[idx]: # reset, new epoch
			self.reset_batch(dataset)
			return None
		else:
			data = (self.word_tensors[idx][self.batch_num[idx]],
					self.emoji_tensors[idx][self.batch_num[idx]])
			self.batch_num[idx] += 1
			return data


	def reset_batch(self, dataset='train'):
		self.batch_num[self.DATA2ID[dataset]] = 0

	def batch_number(self, dataset='train'):
		return self.batch_num[self.DATA2ID[dataset]]

	def batch_count(self, dataset='train'):
		return self.batches[self.DATA2ID[dataset]]

	def sentence2tensor(self, sentence):
		words = cleaned.split()
		tensor = np.zeros(len(words))
		for i, word in enumerate(words):
			try:
				tensor[i] = self.word2id[word]
			except KeyError: # label as unknown if not in vocab
				tensor[i] = 1

		return tensor


class CharLoader(WordLoader):
	def __init__(self, data, batch_size=100, resample=False):
		self.batch_size = batch_size
		try: # to load saved vocab data
			print('loading saved vocabulary and processed data')
			self.char2id, self.id2word, self.emoji2id, self.id2emoji, self.weights = load(os.path.join(self.PATH, '%s_char_vocab.pkl' % data))
			self.char_tensors, self.emoji_tensors = load(os.path.join(self.PATH, '%s_char_tensor.pkl' % data))
			self.max_seq_len = self.char_tensors[0].shape[1]
			self.samples = [tensor.shape[0] for tensor in self.char_tensors]
			self.char_vocab_size = len(self.id2word)
			self.emoji_vocab_size = len(self.emoji2id)
		except (IOError, EOFError) as e:
			print('failed to load, building vocabulary and processed data')
			self._build_vocab(data)

		print("%d chararacters, %d emojis" % (self.char_vocab_size-4, self.emoji_vocab_size))

		if resample:
			self._resample(data)
			self.samples[0] = self.emoji_tensors[0].shape[0]

		# reshape data into batches
		self.batch_num = [0, 0, 0]
		self.batches = list()
		temp_char_tensors = list()
		temp_emoji_tensors = list()
		for idx in xrange(3):
			# remove extra data samples
			offset = self.samples[idx] % self.batch_size
			self.batches.append(self.samples[idx] / self.batch_size)

			char_tensor = self.char_tensors[idx]
			emoji_tensor = self.emoji_tensors[idx]
			if offset != 0:
				char_tensor = char_tensor[:-offset, :, :]
				emoji_tensor = emoji_tensor[:-offset]

			char_tensor = np.vsplit(char_tensor, self.batches[idx])
			emoji_tensor = np.split(emoji_tensor, self.batches[idx])

			temp_char_tensors.append(char_tensor)
			temp_emoji_tensors.append(emoji_tensor)

		self.char_tensors = temp_char_tensors
		self.emoji_tensors = temp_emoji_tensors
	
	# resample training data to even out classes
	def _resample(self, data):
		try:
			print("attempting to load resampled data")
			self.char_tensors[0], self.emoji_tensors[0] = load(os.path.join(self.PATH, '%s_char_resample.pkl' % data))
		except (IOError, EOFError) as e:
			print("failed to load, building resampled data")
			n_resamples = self.samples[0] / self.emoji_vocab_size
			temp_emoji_tensors = list()
			temp_char_tensors = list()
			for i in xrange(self.emoji_vocab_size):
				indices = np.where(self.emoji_tensors[0] == i)[0]
				r_indices = np.random.choice(indices, size=n_resamples, replace=True)
				temp_char_tensors.append(self.char_tensors[0][r_indices, :, :])
				temp_emoji_tensors.append(self.emoji_tensors[0][r_indices])

			shuffled = np.arange(n_resamples * self.emoji_vocab_size)
			np.random.shuffle(shuffled)
			self.char_tensors[0] = np.concatenate(temp_char_tensors, axis=0)[shuffled, :]
			self.emoji_tensors[0] = np.concatenate(temp_emoji_tensors)[shuffled]

			save([self.char_tensors[0], self.emoji_tensors[0]], os.path.join(self.PATH, '%s_char_resample.pkl' % data))


	def _build_vocab(self, data):
		# create vocab and ids from either character or words
		filenames = [
			os.path.join(self.PATH, '%s_train' % data),
			os.path.join(self.PATH, '%s_validation' % data),
			os.path.join(self.PATH, '%s_test' % data)
		]

		# get counts and samples
		self.max_seq_len = 0
		self.max_word_len = 0
		self.samples = list()
		char_counts = Counter()
		emoji_counts = Counter()
		for filename in filenames:
			with open(filename, 'r') as f:
				line_count = 0
				for line in f:
					words = line.split()
					emoji_counts[words[-1]] += 1
					words = words[:-1]
					line_count += 1
					self.max_seq_len = max(self.max_seq_len, len(words))
					for word in words:
						self.max_word_len = max(self.max_word_len, len(word))
						for char in word:
							char_counts[char] += 1

			self.samples.append(line_count)

		self.max_word_len += 2

		# build char vocab, add null, unknown, start, and stop chars
		self.char2id = {'<none>': 0, '<unk>': 1, '<start>': 2, '<stop>': 3}
		self.id2char = ['<none>', '<unk>', '<start>', '<stop>']
		for char, _ in char_counts.most_common():
			self.char2id[char] = len(self.id2char)
			self.id2char.append(char)
		
		# build emoji vocab
		self.emoji2id = dict()
		self.id2emoji = list()
		self.weights = np.zeros(len(emoji_counts))
		self.emoji_vocab_size = len(emoji_counts)
		total_samples = sum(self.samples)
		for emoji, count in emoji_counts.most_common():
			self.emoji2id[emoji] = len(self.id2emoji)
			self.id2emoji.append(emoji)
			self.weights[self.emoji2id[emoji]] = total_samples / count

		self.char_vocab_size = len(self.id2char)
		self.emoji_vocab_size = len(self.id2emoji)

		# create tensor from char ids
		self.char_tensors = list()
		self.emoji_tensors = list()
		for idx, filename in enumerate(filenames):
			char_tensor = np.zeros((self.samples[idx], self.max_seq_len, self.max_word_len), dtype=np.int32)
			emoji_tensor = np.zeros((self.samples[idx]), dtype=np.int32)
			with open(filename, 'r') as f:
				for i, line in enumerate(f):
					words = line.split()
					emoji_tensor[i] = self.emoji2id[words[-1]]
					words = words[:-1]
					for j, word in enumerate(words):
						char_tensor[i,j,0] = 2
						char_tensor[i,j,len(word)+1] = 3
						for k, char in enumerate(word):
							char_tensor[i,j,k+1] = self.char2id[char]

			self.char_tensors.append(char_tensor)
			self.emoji_tensors.append(emoji_tensor)

		# saved vocabulary and processed data
		save([self.char2id, self.id2char, self.emoji2id, self.id2emoji, self.weights], os.path.join(self.PATH, '%s_char_vocab.pkl' % data))
		save([self.char_tensors, self.emoji_tensors], os.path.join(self.PATH, '%s_char_tensor.pkl' % data))


	def next_batch(self, dataset='train'):
		idx = self.DATA2ID[dataset]
		if self.batch_num[idx] >= self.batches[idx]: # reset, new epoch
			self.reset_batch(dataset)
			return None
		else:
			data = (self.char_tensors[idx][self.batch_num[idx]],
					self.emoji_tensors[idx][self.batch_num[idx]])
			self.batch_num[idx] += 1
			return data


	def sentence2tensor(self, sentence):
		words = cleaned.split()
		max_word = 0
		for word in words:
			max_word = max(longest, len(word))

		tensor = np.zeros(len(words), max_word+2)
		for i, word in enumerate(words):
			for j, char in enumerate(word):
				tensor[i,0] = 2
				tensor[i,len(word)+2] = 3
				try:
					tensor[i,j+1] = self.char2id[word]
				except KeyError: # label as unknown if not in vocab
					tensor[i,j+1] = 1

		return tensor

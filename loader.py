import numpy as np
import pandas as pd
import pickle
import string
import os

from collections import Counter
from pandas import read_csv


def save(obj, filename):
	with open(filename, 'wb+') as f:
		pickle.dump(obj, f)

def load(filename):
	with open(filename, 'rb') as f:
		return pickle.load(f)


class WordLoader:
	DATA2ID = {'train': 0, 'validation': 1, 'test': 2}
	PATH = './data'

	def __init__(self, data, batch_size=100, glove=25, resample=False):
		self.batch_size = batch_size
		try: # to load saved vocab data
			print('loading saved vocabulary and processed data')
			self.word2id, self.id2word, self.emoji2id, self.id2emoji, self.weights = load(os.path.join(self.PATH, '%s_word_vocab.pkl' % data))
			self.raw_word_tensors, self.raw_emoji_tensors = load(os.path.join(self.PATH, '%s_word_tensor.pkl' % data))
			self.glove_embed = load(os.path.join(self.PATH, '%s_%d_glove.pkl' % (data, glove)))
			self.max_seq_len = self.raw_word_tensors[0].shape[1]
			self.samples = [tensor.shape[0] for tensor in self.raw_word_tensors]
			self.word_vocab_size = len(self.id2word)
			self.emoji_vocab_size = len(self.emoji2id)
		except (IOError, EOFError) as e:
			print('failed to load, building vocabulary and processed data')
			self._build_vocab(data, glove)

		print("%d words, %d emojis" % (self.word_vocab_size-2, self.emoji_vocab_size))

		if resample:
			self._resample(data)
			self.samples[0] = self.raw_emoji_tensors[0].shape[0]

		# reshape data into batches
		self.batch_num = [0, 0, 0]
		self.batches = list()
		temp_word_tensors = list()
		temp_emoji_tensors = list()
		for idx in range(3):
			# remove extra data samples
			offset = self.samples[idx] % self.batch_size
			self.batches.append(int(self.samples[idx] / self.batch_size))

			word_tensor = self.raw_word_tensors[idx]
			emoji_tensor = self.raw_emoji_tensors[idx]
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
			print("loading resampled data")
			self.raw_word_tensors[0], self.raw_emoji_tensors[0] = load(os.path.join(self.PATH, '%s_resample.pkl' % data))
		except (IOError, EOFError) as e:
			print("failed to load, building resampled data")
			n_resamples = int(self.samples[0] / self.emoji_vocab_size)
			temp_emoji_tensors = list()
			temp_word_tensors = list()
			for i in range(self.emoji_vocab_size):
				indices = np.where(self.raw_emoji_tensors[0] == i)[0]
				print(len(indices))
				r_indices = np.random.choice(indices, size=n_resamples, replace=True)
				temp_word_tensors.append(self.raw_word_tensors[0][r_indices])
				temp_emoji_tensors.append(self.raw_emoji_tensors[0][r_indices])

			shuffled = np.arange(n_resamples * self.emoji_vocab_size)
			np.random.shuffle(shuffled)
			self.raw_word_tensors[0] = np.concatenate(temp_word_tensors, axis=0)[shuffled]
			self.raw_emoji_tensors[0] = np.concatenate(temp_emoji_tensors)[shuffled]

			save([self.raw_word_tensors[0], self.raw_emoji_tensors[0]], os.path.join(self.PATH, '%s_resample.pkl' % data))


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
		for word, counts in word_counts.most_common():
			if counts > 4:
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
		self.raw_word_tensors = list()
		self.raw_emoji_tensors = list()
		for idx, filename in enumerate(filenames):
			word_tensor = np.zeros((self.samples[idx], self.max_seq_len), dtype=np.int64)
			emoji_tensor = np.zeros((self.samples[idx]), dtype=np.int64)
			with open(filename, 'r') as f:
				for i, line in enumerate(f):
					words = line.split()
					emoji_tensor[i] = self.emoji2id[words[-1]]
					words = words[:-1]
					for j, word in enumerate(words):
						try:
							word_tensor[i, j] = self.word2id[word]
						except KeyError:
							word_tensor[i, j] = 1 # unknown

			self.raw_word_tensors.append(word_tensor)
			self.raw_emoji_tensors.append(emoji_tensor)

		# saved vocabulary and processed data
		save([self.word2id, self.id2word, self.emoji2id, self.id2emoji, self.weights], os.path.join(self.PATH, '%s_word_vocab.pkl' % data))
		save([self.raw_word_tensors, self.raw_emoji_tensors], os.path.join(self.PATH, '%s_word_tensor.pkl' % data))
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
		idx = self.DATA2ID[dataset]
		self.batch_num[idx] = 0

		# shuffle words
		temp_word_tensors = list()
		temp_emoji_tensors = list()
	
		# remove extra data samples
		offset = self.samples[idx] % self.batch_size
		shuffled = np.arange(self.samples[idx])
		np.random.shuffle(shuffled)
		
		word_tensor = self.raw_word_tensors[idx][shuffled]
		emoji_tensor = self.raw_emoji_tensors[idx][shuffled]
		if offset != 0:
			word_tensor = word_tensor[:-offset, :]
			emoji_tensor = emoji_tensor[:-offset]

		word_tensor = np.vsplit(word_tensor, self.batches[idx])
		emoji_tensor = np.split(emoji_tensor, self.batches[idx])

		temp_word_tensors.append(word_tensor)
		temp_emoji_tensors.append(emoji_tensor)

		self.word_tensors = temp_word_tensors
		self.emoji_tensors = temp_emoji_tensors


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


class CharLoader:
	DATA2ID = {'train': 0, 'validation': 1, 'test': 2}
	PATH = './data'

	def __init__(self, data, batch_size=100, max_seq=0, max_word=0, resample=False):
		self.batch_size = batch_size
		try: # to load saved vocab data
			print('loading saved vocabulary and processed data')
			self.char2id, self.id2char, self.emoji2id, self.id2emoji = load(
					os.path.join(self.PATH, '%d_char_vocab.pkl' % data))
			self.samples, self.max_seq_len, self.max_word_len = load(
					os.path.join(self.PATH, '%d_char_stats.pkl' % data))
			self.emoji_vocab_size = len(self.id2emoji)
			self.char_vocab_size = len(self.id2char)
			self.filenames = [os.path.join(self.PATH, '%d_train' % data),
										  os.path.join(self.PATH, '%d_validation' % data),
										  os.path.join(self.PATH, '%d_test' % data)]
		except (IOError, EOFError) as e:
			print('failed to load, building vocabulary')
			self._build_vocab(data)

		print('%d characters, %d emojis' % (self.char_vocab_size, self.emoji_vocab_size))
		print('longest word %d, longest sequence %d' % (self.max_word_len, self.max_seq_len))

		if resample:
			print('loading resampled data')
			resample_filename = os.path.join(self.PATH, '%s_resample' % data)
			if not os.path.isfile(resample_filename):
				print('failed to load, building resampled data')
				self._resample(data)

			self.filenames[0] = resample_filename
			self.samples[0] = data * int(self.samples[0] / self.emoji_vocab_size)


		if max_seq > 0:
			self.max_seq_len = min(max_seq, self.max_seq_len)

		if max_word > 0:
			self.max_word_len = min(max_word, self.max_word_len)

		self.batches = [int(sample / self.batch_size) for sample in self.samples]
	
	# resample training data to even out classes
	def _resample(self, data):
		with open(self.filenames[0], 'r') as f:
			old_lines = np.array(f.readlines())

		emojis = np.zeros(old_lines.shape[0])
		for i, line in enumerate(old_lines):
			emojis[i] = self.emoji2id[line.split()[-1]]

		n_resamples = int(self.samples[0] / self.emoji_vocab_size)
		new_lines = list()
		for i in range(self.emoji_vocab_size):
			indices = np.where(emojis == i)[0]
			r_indices = np.random.choice(indices, size=n_resamples, replace=True)
			new_lines.append(old_lines[r_indices])

		self.samples[0] = n_resamples * self.emoji_vocab_size
		shuffled = np.arange(self.samples[0])
		np.random.shuffle(shuffled)
		new_lines = np.concatenate(new_lines)[shuffled]

		with open(os.path.join(self.PATH, '%s_resample' % data), 'w+') as f:
			for line in new_lines:
				f.write(line)


	def _build_vocab(self, data):
		self.filenames = filenames = [os.path.join(self.PATH, '%s_train' % data),
									  os.path.join(self.PATH, '%s_validation' % data),
									  os.path.join(self.PATH, '%s_test' % data)]
		
		char_counts = Counter()
		emoji_counts = Counter()
		self.samples = list()
		self.max_seq_len = 0
		self.max_word_len = 0
		for filename in self.filenames:
			with open(filename, 'r') as f:
				line_count = 0
				for line in f:
					line_count += 1
					words = line.split()
					emoji_counts[words[-1]] += 1
					words = words[:-1]
					self.max_seq_len = max(self.max_seq_len, len(words))
					for word in words:
						self.max_word_len = max(self.max_word_len, len(word))
						for char in word:
							char_counts[char] += 1

				self.samples.append(line_count)
		self.max_word_len

		self.char2id = {'<none>': 0, '<unk>': 1, '<start>': 2, '<stop>': 3}
		self.id2char = ['<none>', '<unk>', '<start>', '<stop>']
		for char, count in char_counts.most_common():
			if count > 1:
				self.char2id[char] = len(self.id2char)
				self.id2char.append(char)

		self.emoji2id = dict()
		self.id2emoji = list()
		for emoji, _ in emoji_counts.most_common():
			self.emoji2id[emoji] = len(self.id2emoji)
			self.id2emoji.append(emoji)

		self.emoji_vocab_size = len(self.id2emoji)
		self.char_vocab_size = len(self.id2char)	

		save([self.char2id, self.id2char, self.emoji2id, self.id2emoji],
				os.path.join(self.PATH, '%s_char_vocab.pkl' % data))
		save([self.samples, self.max_seq_len, self.max_word_len],
				os.path.join(self.PATH, '%s_char_stats.pkl' % data))


	def _shuffle(self):
		with open(self.filenames[0], 'r') as f:
			lines = np.array(f.readlines())

		shuffled = np.arange(lines.shape[0])
		np.random.shuffle(shuffled)
		lines =lines[shuffled]

		self.filenames[0] = 'tmp/5_shuffle'
		with open(self.filenames[0], 'w+') as f:
			for line in lines:
				f.write(line)

	# returns iterator over dataset
	def batch_generator(self, dataset):
		if dataset == 'train':
			self._shuffle()

		with open(self.filenames[self.DATA2ID[dataset]], 'r') as f:
			while True:
				features = np.zeros((self.batch_size, self.max_seq_len, self.max_word_len+2), dtype=np.uint8)
				targets = np.zeros(self.batch_size, dtype=np.uint8)
				for i in range(self.batch_size):
					line = f.__next__()
					if not line:
						return

					tokens = line.split()
					targets[i] = self.emoji2id[tokens[-1]]
					tokens = tokens[:-1]
					for j in range(min(len(tokens), self.max_seq_len)):
						features[i,j,0] = 2
						if (len(tokens[j]) + 1) < self.max_word_len:
							features[i,j,len(tokens[j])+1] = 3

						for k in range(min(len(tokens[j]), self.max_word_len)):
							try:
								features[i,j,k+1] = self.char2id[tokens[j][k]]
							except KeyError:
								features[i,j,k+1] = 1

				yield (features, targets)


	def batch_count(self, dataset):
		return self.batches[self.DATA2ID[dataset]]


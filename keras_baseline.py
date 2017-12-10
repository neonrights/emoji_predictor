import numpy as np
import tensorflow as tf
import os
import re
import glob
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from sklearn.metrics import *
from keras.models import Model, load_model
from keras.initializers import RandomUniform
from keras.layers import *

from loader import WordLoader

name = 'resampled'
emojis = 20
data = WordLoader(data=emojis, batch_size=32, glove=50, resample=True)
start = 0
stop = 100

try:
	print('loading model')
	files = glob.glob('./tmp/%d/%s/epoch*.h5' % (emojis, name))
	if not files:
		raise IOError

	files.sort(key=lambda x: -os.path.getmtime(x))
	newest = files[0]

	model = load_model(newest)
	start = int(re.search('\d+', newest.split('/')[-1])[0])
except (IOError, OSError) as e:
	print('failed to load, building model')
	if not os.path.isdir('./tmp/%d/%s' % (emojis, name)):
		os.mkdir('./tmp/%d/%s' % (emojis, name))

	word_ids = Input(shape=(data.max_seq_len,))
	embedding = Embedding(data.word_vocab_size, 30, input_length=data.max_seq_len)(word_ids)

	convs = list()
	cnn_output = embedding
	for width in range(3,6):
		conv = Conv1D(64, width)(cnn_output)
		pool = MaxPooling1D(pool_size=data.max_seq_len-width+1)(conv)
		flat = Reshape((64,))(pool)
		convs.append(flat)

	cnn_output = Concatenate()(convs)
	cnn_output = Activation('relu')(cnn_output)
	cnn_output = Dropout(0.5)(cnn_output)

	N = 3*64
	neg_init = RandomUniform(-3.0, -1.0)
	hwy_output = cnn_output
	for i in range(0):
		trans_gate = Dense(N, activation='sigmoid', bias_initializer=neg_init)(hwy_output)
		trans_output = Multiply()([trans_gate, Dense(N, activation='relu')(hwy_output)])
		carry_output = Multiply()([Lambda(lambda x: 1 - x)(trans_gate), hwy_output])
		hwy_output = Add()([trans_output, carry_output])
		hwy_output = Dropout(0.5)(hwy_output)

	hidden = cnn_output
	for i in range(0):
		hidden = Dense(100, activation='relu')(hidden)
		hidden = Dropout(0.5)(hidden)

	probs = Dense(emojis, activation='softmax')(hwy_output)

	model = Model(inputs=word_ids, outputs=probs)
	model.compile(optimizer='adam', loss='categorical_crossentropy')

loss = model.evaluate(x=data.raw_word_tensors[1], y=np.eye(emojis)[data.raw_emoji_tensors[1]])
predicted = model.predict(x=data.raw_word_tensors[1])
predicted = np.argmax(predicted, axis=1)
print('Initial Validation loss and F1 %2.6f %2.6f' % (loss,
		f1_score(data.raw_emoji_tensors[1], predicted, average='weighted')))

check = int(data.batch_count('train') / 1000) * 100
for epoch in range(start, stop):
	data.reset_batch('train')
	for i in range(data.batch_count('train')):
		batch = data.next_batch('train')
		loss = model.train_on_batch(x=batch[0], y=np.eye(emojis)[batch[1]])
		if (i+1) % check == 0:
			print("epoch %d: %d/%d test loss %2.6f" % (epoch+1, i+1, data.batch_count('train'), loss))

	loss = model.evaluate(x=data.raw_word_tensors[1], y=np.eye(emojis)[data.raw_emoji_tensors[1]])
	predicted = model.predict(x=data.raw_word_tensors[1])
	predicted = np.argmax(predicted, axis=1)
	print('Epoch %d Validation loss and F1 %2.6f %2.6f' % (epoch+1,	loss,
			f1_score(data.raw_emoji_tensors[1], predicted, average='weighted')))

	predicted = model.predict(x=data.raw_word_tensors[2])
	predicted = np.argmax(predicted, axis=1)

	print('Accuracy %2.6f' % accuracy_score(data.raw_emoji_tensors[2], predicted))
	print('Recall %2.6f' % recall_score(data.raw_emoji_tensors[2], predicted, average='weighted'))
	print('Precision %2.6f' % precision_score(data.raw_emoji_tensors[2], predicted, average='weighted'))
	print('F1 %2.6f' % f1_score(data.raw_emoji_tensors[2], predicted, average='weighted'))

	conf_mat = confusion_matrix(data.raw_emoji_tensors[2], predicted, labels=np.arange(emojis))
	fig, ax = plt.subplots()
	fig.suptitle('Confusion Matrix')
	plt.xlabel('Predicted Label')
	plt.ylabel('True Label')
	cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)
	for i in range(emojis):
		for j in range(emojis):
			ax.text(i, j, str(conf_mat[j,i]), va='center', ha='center')

	fig.colorbar(cax)
	plt.savefig('./tmp/%d/%s/confusion%d.png' % (emojis, name, epoch+1), bbox_inches='tight', transparent=True, pad_inches=0)
	plt.close(fig)

	model.save('./tmp/%d/%s/epoch%s.h5' % (emojis, name, epoch+1))

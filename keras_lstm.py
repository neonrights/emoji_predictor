import numpy as np
import tensorflow as tf
import os
import re
import glob
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, confusion_matrix
from keras.models import Model, load_model
from keras.layers import *
from keras import optimizers
from keras.initializers import RandomUniform

from loader import CharLoader

name = 'char_lstm'
emojis = 5
epochs = 100
data = CharLoader(data=emojis, batch_size=32, max_word=30, resample=True)
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
	print('failed to load, initializing model')
	if not os.path.isdir('./tmp/%d/%s' % (emojis, name)):
		os.mkdir('./tmp/%d/%s' % (emojis, name))

	# embeddings
	char_ids = Input((data.max_seq_len, data.max_word_len+2))
	char_embeddings = TimeDistributed(Embedding(data.char_vocab_size, 15))(char_ids)

	# convolutions
	convs = list()
	N = 0
	for i in range(1,8):
		N += 25*i
		conv = TimeDistributed(Conv1D(25*i, i))(char_embeddings)
		pool = TimeDistributed(MaxPooling1D(pool_size=data.max_word_len-i+1))(conv)
		flat = TimeDistributed(Reshape((25*i,)))(pool)
		convs.append(flat)

	cnn_output = Concatenate(axis=2)(convs)

	# highway
	neg_init = RandomUniform(-3.0, -1.0)
	hwy_output = cnn_output
	for i in range(1):
		trans_gate = Dense(N, activation='sigmoid', bias_initializer=neg_init)(hwy_output)
		trans_output = Multiply()([trans_gate, Dense(N, activation='relu')(hwy_output)])
		carry_output = Multiply()([Lambda(lambda x: 1 - x)(trans_gate), hwy_output])
		hwy_output = Add()([trans_output, carry_output])

	# lstm
	#lstm = BiDirectional(LSTM(300, dropout=0.5, recurrent_dropout=0.5))(hwy_output)
	lstm = LSTM(300, dropout=0.5)(hwy_output)
	probs = Dense(data.emoji_vocab_size, activation='softmax')(lstm)

	model = Model(inputs=char_ids, outputs=probs)
	model.compile(optimizer=optimizers.Adadelta(lr=1, clipnorm=5.0), loss='categorical_crossentropy')

# calculate initial validation loss and F1
batch_count = data.batch_count('validation')
loss = 0
predicted = list()
true_vals = list()
for i, batch in enumerate(data.batch_generator('validation')):
	loss += model.test_on_batch(x=batch[0], y=np.eye(emojis)[batch[1]])
	batch_predict = model.predict(x=batch[0])
	predicted.append(np.argmax(batch_predict, axis=1))
	true_vals.append(batch[1])

loss /= batch_count
predicted = np.concatenate(predicted)
true_vals = np.concatenate(true_vals)

print('Initial Validation loss and F1 %2.6f %2.6f' % (loss,
		f1_score(true_vals, predicted, average='weighted')))

# train model
check = int(data.batch_count('train') / 1000) * 100
for epoch in range(start, stop):
	for i, batch in enumerate(data.batch_generator('train')):
		loss = model.train_on_batch(x=batch[0], y=np.eye(emojis)[batch[1]])
		if (i+1) % check == 0:
			print("epoch %d: %d/%d test loss %2.6f" % (epoch+1, i+1, data.batch_count('train'), loss))

	batch_count = data.batch_count('validation')
	loss = 0
	predicted = list()
	true_vals = list()
	for i, batch in enumerate(data.batch_generator('validation')):
		loss += model.test_on_batch(x=batch[0], y=np.eye(emojis)[batch[1]])
		batch_predict = model.predict(x=batch[0])
		predicted.append(np.argmax(batch_predict, axis=1))
		true_vals.append(batch[1])

	loss /= batch_count
	predicted = np.concatenate(predicted)
	true_vals = np.concatenate(true_vals)
	
	print('Epoch %d Validation loss and F1 %2.6f %2.6f' % (epoch+1,	loss,
			f1_score(true_vals, predicted, average='weighted')))

	batch_count = data.batch_count('test')
	predicted = list()
	true_vals = list()
	for i, batch in enumerate(data.batch_generator('test')):
		batch_predict = model.predict(x=batch[0])
		predicted.append(np.argmax(batch_predict, axis=1))
		true_vals.append(batch[1])

	predicted = np.concatenate(predicted)
	true_vals = np.concatenate(true_vals)

	print('Accuracy %2.6f' % accuracy_score(true_vals, predicted))
	print('Recall %2.6f' % recall_score(true_vals, predicted, average='weighted'))
	print('Precision %2.6f' % precision_score(true_vals, predicted, average='weighted'))
	print('F1 %2.6f' % f1_score(true_vals, predicted, average='weighted'))

	conf_mat = confusion_matrix(true_vals, predicted, labels=np.arange(emojis))
	fig, ax = plt.subplots()
	fig.suptitle('Confusion Matrix')
	plt.xlabel('Predicted Label')
	plt.ylabel('True Label')
	cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)
	for i in range(emojis):
		for j in range(emojis):
			ax.text(i, j, str(conf_mat[j,i]), va='center', ha='center')

	fig.colorbar(cax)
	fig.savefig('./tmp/%d/%s/lstm%d.png' % (emojis, name, epoch+1))
	plt.close(fig)

	model.save('./tmp/%d/%s/epoch%s.h5' % (emojis, name, epoch+1))


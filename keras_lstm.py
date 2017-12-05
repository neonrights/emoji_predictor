import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from sklearn.metrics import *
from keras.models import Model
from keras.layers import *

from loader import CharLoader

K = 20
epochs = 100
data = CharLoader(data=K, max_word=50, resample=True)

# embeddings
char_ids = Input((data.max_seq_len, data.max_word_len))
char_embeddings = TimeDistributed(Embedding(data.char_vocab_size, 15))(char_ids)

# convolutions
convs = list()
N = 0
for i in range(8):
	N += 25*i
	conv = TimeDistributed(Conv1D(25*i, i))(char_embeddings)
	pool = TimeDistributed(MaxPooling1D(pool_size=data.max_seq_len-i+1))(conv)
	flat = TimeDistributed(Reshape((25*i,)))(pool)
	convs.append(flat)

cnn_output = Concatenate(axis=2)(convs)
cnn_output = Dropout(0.5)(cnn_output)

# highway
trans_gate = TimeDistributed(Dense(N), activation='sigmoid')(cnn_output)
trans_output = Multiply()([trans_gate, Dense(N, activation='relu')(cnn_output)])
carry_output = Multiply()([Lambda(lambda x: 1 - x)(trans_gate), cnn_output])
hwy_output = Add()([trans_output, carry_output])
hwy_output = Dropout(0.5)(hwy_output)

# lstm
lstm = LSTM(300, dropout=0.5)(hwy_output)
probs = Dense(data.emoji_vocab_size, activation='softmax')(lstm)

model = Model(inputs=char_ids, outputs=probs)
model.compile(optimizer='adam', loss='categorical_crossentropy')

# calculate initial validation loss and F1
batch_count = data.batch_count('validation')
predicted = np.zeros(batch_count * data.batch_size)
true_vals = np.zeros(batch_count * data.batch_size)
for i, batch in enumerate(data.batch_generator('validation')):
	batch_predict = model.predict(x=batch[0])
	predicted[(batch_count*i):((i+1)*batch_count)] = np.argmax(batch_predict, axis=1)
	true_vals[(batch_count*i):((i+1)*batch_count)] = batch[1]

print('Initial Validation loss and F1 %2.6f %2.6f' % (loss,
		f1_score(true_vals, predicted, average='weighted')))

# train model
check = int(data.batch_count('train') / 1000) * 100
for epoch in range(epochs):
	for i, batch in enumerate(data.batch_generator('train')):
		loss = model.train_on_batch(x=batch[0], y=np.eye(K)[batch[1]])
		if (i+1) % check == 0:
			print("epoch %d: %d/%d test loss %2.6f" % (epoch+1, i+1, data.batch_count('train'), loss))

	batch_count = data.batch_count('validation')
	predicted = np.zeros(batch_count * data.batch_size)
	true_vals = np.zeros(batch_count * data.batch_size)
	for i, batch in enumerate(data.batch_generator('validation')):
		batch_predict = model.predict(x=batch[0])
		predicted[(batch_count*i):((i+1)*batch_count)] = np.argmax(batch_predict, axis=1)
		true_vals[(batch_count*i):((i+1)*batch_count)] = batch[1]
	
	print('Epoch %d Validation loss and F1 %2.6f %2.6f' % (epoch+1,	loss,
			f1_score(true_vals, predicted, average='weighted')))

	batch_count = data.batch_count('test')
	predicted = np.zeros(batch_count * data.batch_size)
	true_vals = np.zeros(batch_count * data.batch_size)
	for i, batch in enumerate(data.batch_generator('test')):
		batch_predict = model.predict(x=batch[0])
		predicted[(batch_count*i):((i+1)*batch_count)] = np.argmax(batch_predict, axis=1)
		true_vals[(batch_count*i):((i+1)*batch_count)] = batch[1]

	print('Accuracy %2.6f' % accuracy_score(, predicted))
	print('Recall %2.6f' % recall_score(true_vals, predicted, average='weighted'))
	print('Precision %2.6f' % precision_score(true_vals, predicted, average='weighted'))
	print('F1 %2.6f' % f1_score(true_vals, predicted, average='weighted'))

	conf_mat = confusion_matrix(true_vals, predicted, labels=np.arange(K))
	fig, ax = plt.subplots()
	fig.suptitle('Confusion Matrix')
	plt.xlabel('Predicted Label')
	plt.ylabel('True Label')
	cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)
	for i in range(K):
		for j in range(K):
			ax.text(i, j, str(conf_mat[j,i]), va='center', ha='center')

	fig.colorbar(cax)
	fig.savefig('confusion%d.png' % epoch)
	plt.close(fig)


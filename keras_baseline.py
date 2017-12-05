import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from sklearn.metrics import *
from keras.models import Model
from keras.layers import *
from keras import backend

from loader import WordLoader

K = 20
epochs = 100
data = WordLoader(data=K, glove=50, resample=True)

word_ids = Input(shape=(data.max_seq_len,))
embedding = Embedding(data.word_vocab_size, 50, input_length=data.max_seq_len)(word_ids)

convs = list()
for i in range(3,6):
	conv = Conv1D(64, i)(embedding)
	pool = MaxPooling1D(pool_size=data.max_seq_len-i+1)(conv)
	flat = Reshape((64,))(pool)
	convs.append(flat)

cnn_output = Concatenate()(convs)
cnn_relu = Activation('relu')(cnn_output)
cnn_dropout = Dropout(0.5)(cnn_relu)

N = 3*64
trans_gate = Dense(N, activation='sigmoid')(cnn_dropout)
trans_output = Multiply()([trans_gate, Dense(N, activation='relu')(cnn_dropout)])
carry_output = Multiply()([Lambda(lambda x: 1 - x)(trans_gate), cnn_dropout])
hwy_output = Add()([trans_output, carry_output])
hwy_dropout = Dropout(0.5)(hwy_output)

probs = Dense(K, activation='softmax')(hwy_dropout)

model = Model(inputs=word_ids, outputs=probs)
model.compile(optimizer='adam', loss='categorical_crossentropy')

loss = model.evaluate(x=data.raw_word_tensors[1], y=np.eye(K)[data.raw_emoji_tensors[1]])
predicted = model.predict(x=data.raw_word_tensors[1])
predicted = np.argmax(predicted, axis=1)
print('Initial Validation loss and F1 %2.6f %2.6f' % (loss,
		f1_score(data.raw_emoji_tensors[1], predicted, average='weighted')))

check = int(data.batch_count('train') / 1000) * 100
for epoch in range(epochs):
	data.reset_batch('train')
	for i in range(data.batch_count('train')):
		batch = data.next_batch('train')
		loss = model.train_on_batch(x=batch[0], y=np.eye(K)[batch[1]])
		if (i+1) % check == 0:
			print("epoch %d: %d/%d test loss %2.6f" % (epoch+1, i+1, data.batch_count('train'), loss))

	loss = model.evaluate(x=data.raw_word_tensors[1], y=np.eye(K)[data.raw_emoji_tensors[1]])
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

	conf_mat = confusion_matrix(data.raw_emoji_tensors[2], predicted, labels=np.arange(K))
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


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from emoji_cnn import EmojiCNN

fig = plt.figure()
fig.suptitle("Validation Loss vs Epoch per Batch Size")
plt.xlabel("epoch")
plt.ylabel("loss")

batch_sizes = [10, 30, 100, 300]
handles = list()
widths = np.arange(1,8)
filters = [25*width for width in widths]
for batch_size in batch_sizes:
	with tf.Session() as sess:
		model = EmojiCNN(sess,
					data='5',
					batch_size=batch_size,
					name="test_model_b%d" % batch_size,
					embed_dim=50,
					kernel_widths=widths,
					kernel_filters=filters,
					layers=[300],
					restore=True)
		_, valid_loss = model.run(epochs=10)
		handle, _ = plt.plot(valid_loss)
		handles.append(handle)

plt.legend(handles, batch_sizes)
fig.savefig("batch_size.png")
plt.close(fig)


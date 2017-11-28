import tensorflow as tf
import numpy as np

from emoji_cnn import EmojiCNN

with tf.Session() as sess:
	kernels = np.arange(1,8)
	model = EmojiCNN(sess,
                data='5',
                batch_size=100,
                name="test-model",
                embed_dim=50,
                kernels=[(width, 25*width) for width in kernels],
                layers=[300])
	model.run(epochs=100)


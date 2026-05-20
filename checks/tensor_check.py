import time

import tensorflow as tf

SIZE = 25000  # start smaller



with tf.device('/CPU:0'):
    x = tf.random.normal((SIZE, SIZE), dtype=tf.float32)
    y = tf.random.normal((SIZE, SIZE), dtype=tf.float32)
    start = time.time()
    z = tf.matmul(x, y)
    # force execution
    z.numpy()

    print(f"CPU Time: {time.time() - start} seconds")

with tf.device('/GPU:0'):
    x = tf.random.normal((SIZE, SIZE), dtype=tf.float32)
    y = tf.random.normal((SIZE, SIZE), dtype=tf.float32)

    start = time.time()

    z = tf.matmul(x, y)

    # force execution
    z.numpy()

    print(tf.config.list_physical_devices('GPU'))
    print(f"GPU Time: {time.time() - start} seconds")
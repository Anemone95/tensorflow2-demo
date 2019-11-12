#!/usr/bin/env python3
"""

:Author: Anemone Xu
:Email: anemone95@qq.com
:copyright: (c) 2019 by Anemone Xu.
:license: Apache 2.0, see LICENSE for more details.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt

# config = tf.ConfigProto(allow_soft_placement=True)
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)

# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     print('gpu', gpu)
#     tf.config.experimental.set_memory_growth(gpu, True)
#     print('memory growth:' , tf.config.experimental.get_memory_growth(gpu))

def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string], '')
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()


if __name__ == '__main__':
    dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True,
                              as_supervised=True)
    train_dataset, test_dataset = dataset['train'], dataset['test']
    # text编码器
    encoder = info.features['text'].encoder

    BUFFER_SIZE = 10000
    BATCH_SIZE = 64
    # 随机打乱元素
    train_dataset = train_dataset.shuffle(BUFFER_SIZE)

    # 将字符串填充到最长字符串长度
    train_dataset = train_dataset.padded_batch(BATCH_SIZE, train_dataset.output_shapes)
    test_dataset = test_dataset.padded_batch(BATCH_SIZE, test_dataset.output_shapes)

    # 初始化一个BLSTM
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(encoder.vocab_size, 64),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),  # BLSTM
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # 配置BLSTM
    model.compile(loss='binary_crossentropy',  # 损失函数
                  optimizer=tf.keras.optimizers.Adam(1e-4),  # 优化器
                  metrics=['accuracy'])

    # 训练
    history = model.fit(train_dataset, epochs=5,
                        validation_data=test_dataset,
                        validation_steps=30)

    # 测试
    test_loss, test_acc = model.evaluate(test_dataset)

    print('Test Loss: {}'.format(test_loss))
    print('Test Accuracy: {}'.format(test_acc))

#!/usr/bin/env python3
"""

:Author: Anemone Xu
:Email: anemone95@qq.com
:copyright: (c) 2019 by Anemone Xu.
:license: Apache 2.0, see LICENSE for more details.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import os

import tensorflow as tf
import tensorflow_datasets as tfds


def labeler(example, index):
    """
    打标签
    :param example:
    :param index:
    :return:
    """
    return example, tf.cast(index, tf.int64)


def encode(text_tensor, label):
    """
    编码器
    :param text_tensor:
    :param label:
    :return:
    """
    encoded_text = encoder.encode(text_tensor.numpy())
    return encoded_text, label


def pad_to_size(vec, size):
    zeros = [0] * (size - len(vec))
    vec.extend(zeros)
    return vec


def sample_predict(sentence, encoder, model, len):
    encoded_sample_pred_text = encoder.encode(sentence)

    encoded_sample_pred_text = pad_to_size(encoded_sample_pred_text, len)
    encoded_sample_pred_text = tf.cast(encoded_sample_pred_text, tf.int64)
    predictions = model.predict(tf.expand_dims(encoded_sample_pred_text, 0))

    return predictions


def encode_map_fn(text, label):
    return tf.py_function(encode, inp=[text, label], Tout=(tf.int64, tf.int64))


if __name__ == '__main__':
    # 从数据文件夹中读取文件(一个文件相当于一个数据集)，文件名绑定对应标签
    data_dir = 'data'
    labeled_data_sets = []
    for file_name in os.listdir('data'):
        label = int(file_name.split("_")[1])
        lines_dataset = tf.data.TextLineDataset(os.path.join(data_dir, file_name))
        # 打标签, 因为每个文件可以有多条数据，所以要用map
        labeled_dataset = lines_dataset.map(lambda ex: labeler(ex, label))
        labeled_data_sets.append(labeled_dataset)

    # 将所有数据合并到一个数据集中
    all_labeled_data = labeled_data_sets[0]
    for labeled_dataset in labeled_data_sets[1:]:
        all_labeled_data = all_labeled_data.concatenate(labeled_dataset)

    # 打印部分数据
    for ex in all_labeled_data.take(5):
        print(ex)

    # 打乱数据
    BUFFER_SIZE = 1000  # 要大于数据数
    all_labeled_data = all_labeled_data.shuffle(BUFFER_SIZE, reshuffle_each_iteration=False)

    # 建立单词集合
    tokenizer = tfds.features.text.Tokenizer()
    vocabulary_set = set()
    for text_tensor, _ in all_labeled_data:
        some_tokens = tokenizer.tokenize(text_tensor.numpy())
        vocabulary_set.update(some_tokens)
    print("Vocabulary set:")
    print(vocabulary_set)

    # 对字符串编码
    encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)

    example_text = next(iter(all_labeled_data))[0].numpy()
    encoded_example = encoder.encode(example_text)
    print("Before Encode:", example_text)
    print("Encode:", encoded_example)
    print("Decode:", encoder.decode(encoded_example))

    # 打包编码器为py_function然后调用map
    all_encoded_data = all_labeled_data.map(encode_map_fn)

    # 对数据分组（之后按组计算损失函数），并且填充文本至固定长度，这时vocabsize=len（vocabulary_set）+1
    BATCH_SIZE = 64  # BATCH_SIZE>epoch*epoches
    TAKE_SIZE = 40
    all_encoded_data = all_encoded_data.repeat(5000)
    train_data = all_encoded_data.skip(TAKE_SIZE).shuffle(BUFFER_SIZE)
    train_data = train_data.padded_batch(BATCH_SIZE, padded_shapes=([-1], []))
    # 获取补全后的长度
    FIXED_LENGTH=next(iter(train_data))[0].shape[1]

    test_data = all_encoded_data.take(TAKE_SIZE)
    test_data = test_data.padded_batch(BATCH_SIZE, padded_shapes=([-1], []))

    sample_text, sample_labels = next(iter(test_data))
    print(sample_text[0], sample_labels[0])

    # 初始化一个BLSTM
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(encoder.vocab_size + 1, 64),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),  # BLSTM
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # 标签个数
    ])

    # 配置BLSTM
    model.compile(loss='binary_crossentropy',  # 损失函数
                  optimizer='adam',  # 优化器
                  metrics=['accuracy'])

    # 训练
    if BUFFER_SIZE % BATCH_SIZE != 0:
        parallel_steps = BUFFER_SIZE // BATCH_SIZE + 1
    else:
        parallel_steps = BUFFER_SIZE // BATCH_SIZE
    history = model.fit(train_data, epochs=2, steps_per_epoch=parallel_steps)

    # 测试
    test_loss, test_acc = model.evaluate(test_data)

    print('Test Loss: {}'.format(test_loss))
    print('Test Accuracy: {}'.format(test_acc))

    print(sample_predict("You are wrong!", encoder, model, FIXED_LENGTH))


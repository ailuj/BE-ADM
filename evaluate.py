import os
import argparse
import json
from collections import namedtuple
import numpy as np
import tensorflow as tf
import model.data as data
import model.model as m
import model.evaluate as e
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt

import random


sns.set(rc={'figure.figsize':(11.7,8.27)})
palette = sns.color_palette("bright", 10)


def evaluate(model, dataset, params):
    with tf.Session(config=tf.ConfigProto(
        inter_op_parallelism_threads=params.num_cores,
        intra_op_parallelism_threads=params.num_cores,
        gpu_options=tf.GPUOptions(allow_growth=True)
    )) as session:
        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()

        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(params.model)
        saver.restore(session, ckpt.model_checkpoint_path)

        print('computing vectors...')

        validation_labels = np.array(
            [[y] for y, _ in dataset.rows('validation', num_epochs=1)]
        )
        training_labels = np.array(
            [[y] for y, _ in dataset.rows('training', num_epochs=1)]
        )
        training_labels = np.concatenate(
            (training_labels, validation_labels),
            0
        )
        test_labels = np.array(
            [[y] for y, _ in dataset.rows('test', num_epochs=1)]
        )

        validation_vectors = m.vectors(
            model,
            dataset.batches('validation', params.batch_size, num_epochs=1),
            session
        )
        training_vectors = m.vectors(
            model,
            dataset.batches('training', params.batch_size, num_epochs=1),
            session
        )
        training_vectors = np.concatenate(
            (training_vectors, validation_vectors),
            0
        )
        test_vectors = m.vectors(
            model,
            dataset.batches('test', params.batch_size, num_epochs=1),
            session
        )

        print('evaluating...')

        print("TRAINING VECTORS")
        print(training_vectors[0])
        print("TRAINING LABELS")
        print(training_labels)

        recall_values = [0.0001, 0.0002, 0.0005, 0.002, 0.01, 0.05, 0.2]
        results = e.evaluate(
            training_vectors,
            test_vectors,
            training_labels,
            test_labels,
            recall_values
        )
        for i, r in enumerate(recall_values):
            print('precision @ {}: {}'.format(r, results[i]))

def plot_tsne(model, dataset, params):
    with tf.Session(config=tf.ConfigProto(
            inter_op_parallelism_threads=params.num_cores,
            intra_op_parallelism_threads=params.num_cores,
            gpu_options=tf.GPUOptions(allow_growth=True)
    )) as session:
        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()

        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(params.model)
        saver.restore(session, ckpt.model_checkpoint_path)

        print('computing vectors...')

        validation_labels = np.array(
            [[y] for y, _ in dataset.rows('validation', num_epochs=1)]
        )
        training_labels = np.array(
            [[y] for y, _ in dataset.rows('training', num_epochs=1)]
        )
        training_labels = np.concatenate(
            (training_labels, validation_labels),
            0
        )
        test_labels = np.array(
            [[y] for y, _ in dataset.rows('test', num_epochs=1)]
        )

        validation_vectors = m.vectors(
            model,
            dataset.batches('validation', params.batch_size, num_epochs=1),
            session
        )
        training_vectors = m.vectors(
            model,
            dataset.batches('training', params.batch_size, num_epochs=1),
            session
        )
        training_vectors = np.concatenate(
            (training_vectors, validation_vectors),
            0
        )
        test_vectors = m.vectors(
            model,
            dataset.batches('test', params.batch_size, num_epochs=1),
            session
        )
        print("TEST VECTORS: " + str(test_vectors.shape))
        print(test_vectors)
        print("TEST LABELS: " + str(test_labels.shape))
        print(training_labels[:100])

        #data_X = pd.DataFrame(columns=["test_vectors", "test_labels"])
        #data_X["test_vectors"] = test_vectors
        #data_X["test_labels"] = test_labels



        tsne = TSNE(perplexity=40)
        tsne_obj = tsne.fit_transform(test_vectors)
        tsne_df = pd.DataFrame({'X': tsne_obj[1:, 0],
                                'Y': tsne_obj[1:, 1],
                                'test_labels': test_labels.flatten()[1:]
                                })
        #print(tsne_df.head(20))
        sns.scatterplot(x="X", y="Y",
                        hue="test_labels",
                        palette=sns.color_palette("deep", 10),
                        legend='full',
                        data=tsne_df)
        plt.show()

def topic_word_dist(model, dataset, params, num_words):
    with tf.Session(config=tf.ConfigProto(
            inter_op_parallelism_threads=params.num_cores,
            intra_op_parallelism_threads=params.num_cores,
            gpu_options=tf.GPUOptions(allow_growth=True)
    )) as session:
        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()

        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(params.model)
        saver.restore(session, ckpt.model_checkpoint_path)
        print(tf.get_default_graph())
        w = tf.get_default_graph().get_tensor_by_name("discriminator/h0/w:0")
        w = w.eval().transpose()
        hidden_index = random.sample(range(0, 49), 12)
        with open(params.vocab, 'r') as f:
            vocab = [w.strip() for w in f.readlines()]
        vocab_to_id = dict(zip(range(len(vocab)), vocab))
        topics = []
        for item in w:
            top_words = []
            top_weights = item.argsort()[-num_words:][::-1]
            for weight in top_weights:
                top_words.append(vocab_to_id[weight])
            topics.append(top_words)

        for row in topics:
            print(row)
        #for l in hidden_index:
         #   top_weights = w[l].argsort()[-10:][::-1]

    # print(session.run(["discriminator/h0/w"]))

def evaluate_loss(model, dataset, params):
    with tf.Session(config=tf.ConfigProto(
            inter_op_parallelism_threads=params.num_cores,
            intra_op_parallelism_threads=params.num_cores,
            gpu_options=tf.GPUOptions(allow_growth=True)
    )) as session:
        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()

        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(params.model)
        saver.restore(session, ckpt.model_checkpoint_path)

        print('evaluating loss')
        avg_loss, loss_list = m.loss(
            model,
            dataset.batches('validation', params.batch_size, num_epochs=1),
            session
        )
        print('loss: {}'.format(avg_loss))
        plt.plot(loss_list)
        plt.show()





def main(args):
    with open(os.path.join(args.model, 'params.json'), 'r') as f:
        params = json.loads(f.read())
    params.update(vars(args))
    params = namedtuple('Params', params.keys())(*params.values())

    dataset = data.Dataset(args.dataset)
    x = tf.placeholder(tf.float32, shape=(None, params.vocab_size), name='x')
    z = tf.placeholder(tf.float32, shape=(None, params.z_dim), name='z')
    mask = tf.placeholder(
        tf.float32,
        shape=(None, params.vocab_size),
        name='mask'
    )
    model = m.ADM(x, z, mask, params)
    #evaluate(model, dataset, params)
    #plot_tsne(model, dataset, params)
    #topic_word_dist(model, dataset, params, 10)
    evaluate_loss(model, dataset, params)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='model18',
                        help='path to model output directory')
    parser.add_argument('--dataset', type=str, default='data/risks_preprocessed',
                        help='path to the input dataset')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='the batch size')
    parser.add_argument('--num-cores', type=int, default=1,
                        help='the number of CPU cores to use')
    parser.add_argument('--wasserstein', type=bool, default=True,
                        help='whether to use wasserstein loss')
    parser.add_argument('--vocab', type=str, default="data/risks_2000.vocab",
                        help='path to the vocab')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())

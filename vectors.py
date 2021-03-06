import os
import argparse
import json
from collections import namedtuple
import numpy as np
import tensorflow as tf
import model.data as data
import model.model as m


def save_vectors(vectors, collection, model_dir):
    output_path = os.path.join(model_dir, '{}.txt'.format(collection))
    print('saving vectors to: {}'.format(output_path))
    np.savetxt(output_path, vectors)


def vectors(model, dataset, params):
    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=params.gpu_memory
    )
    with tf.Session(config=tf.ConfigProto(
        inter_op_parallelism_threads=params.num_cores,
        intra_op_parallelism_threads=params.num_cores,
        gpu_options=gpu_options
    )) as session:
        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()

        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(params.model)
        saver.restore(session, ckpt.model_checkpoint_path)

        for collection in dataset.collections:
            save_vectors(
                m.vectors(
                    model,
                    dataset.batches(
                        collection,
                        params.batch_size,
                        num_epochs=1
                    ),
                    session
                ),
                collection,
                params.model
            )


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
    vectors(model, dataset, params)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="model12",
                        help='path to model output directory')
    parser.add_argument('--dataset', type=str, default="data/risks_preprocessed",
                        help='path to the input dataset')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='the batch size')
    parser.add_argument('--num-cores', type=int, default=1,
                        help='the number of CPU cores to use')
    parser.add_argument('--gpu-memory', type=float, default=0.25,
                        help='the ammount of GPU memory used')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())

import os
import argparse
import json
import numpy as np
import tensorflow as tf
import model.data as data
import model.model as m
import model.evaluate as e
import time

import matplotlib.pyplot as plt

seed = 42
np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)


def update(model, x, opt, loss, params, session):
    z = np.random.normal(0, 1, (params.batch_size, params.z_dim))
    mask = np.ones((params.batch_size, params.vocab_size)) * np.random.choice(
        2,
        params.vocab_size,
        p=[params.noise, 1.0 - params.noise]
    )
    loss, _ = session.run([loss, opt], feed_dict={
        model.x: x,
        model.z: z,
        model.mask: mask
    })
    return loss


def train(model, dataset, params, wasserstein=False):
    log_dir = os.path.join(params.model, 'logs')
    model_dir = os.path.join(params.model, 'model')

    with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
        inter_op_parallelism_threads=params.num_cores,
        intra_op_parallelism_threads=params.num_cores,
        gpu_options=tf.compat.v1.GPUOptions(allow_growth=True)
    )) as session:
        avg_d_loss = tf.compat.v1.placeholder(tf.float32, [], 'd_loss_ph')
        tf.compat.v1.summary.scalar('d_loss', avg_d_loss)
        avg_g_loss = tf.compat.v1.placeholder(tf.float32, [], 'g_loss_ph')
        tf.compat.v1.summary.scalar('g_loss', avg_g_loss)

        validation = tf.compat.v1.placeholder(tf.float32, [], 'validation_ph')
        tf.compat.v1.summary.scalar('validation', validation)

        summary_writer = tf.compat.v1.summary.FileWriter(log_dir, session.graph)
        summaries = tf.compat.v1.summary.merge_all()
        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())

        tf.compat.v1.local_variables_initializer().run()
        tf.compat.v1.global_variables_initializer().run()

        d_losses = []
        g_losses = []
        all_losses_d = []
        all_losses_g = []

        # This currently streams from disk. You set num_epochs=1 and
        # wrap this call with something like itertools.cycle to keep
        # this data in memory.
        training_data = dataset.batches('training', params.batch_size)
        best_val = 0.0
        training_labels = np.array(
            [[y] for y, _ in dataset.rows('training', num_epochs=1)]
        )
        validation_labels = np.array(
            [[y] for y, _ in dataset.rows('validation', num_epochs=1)]
        )

        for step in range(params.num_steps + 1):
            _, x = next(training_data)
            # update discriminator
            d_losses.append(update(
                model,
                x,
                model.d_opt,
                model.d_loss,
                params,
                session,
            ))

            # update generator
            g_losses.append(update(
                model,
                x,
                model.g_opt,
                model.g_loss,
                params,
                session,
            ))


            if step % params.log_every == 0:
                print('{}: {:.6f}\t{:.6f}'.format(
                    step,
                    d_losses[-1],
                    g_losses[-1]
                ))
                all_losses_d.extend(d_losses)
                all_losses_g.extend(g_losses)

            if step and (step % params.save_every) == 0:
                validation_vectors = m.vectors(
                    model,
                    dataset.batches(
                        'validation',
                        params.batch_size,
                        num_epochs=1
                    ),
                    session
                )
                training_vectors = m.vectors(
                    model,
                    dataset.batches(
                        'training',
                        params.batch_size,
                        num_epochs=1
                    ),
                    session
                )
                val = e.evaluate(
                    training_vectors,
                    validation_vectors,
                    training_labels,
                    validation_labels
                )[0]
                print('validation: {:.3f} (best: {:.3f})'.format(
                    val,
                    best_val or 0.0
                ))

                #if val > best_val:
                best_val = val
                print('saving: {}'.format(model_dir))
                saver.save(session, model_dir, global_step=step)

                summary, = session.run([summaries], feed_dict={
                    model.x: x,
                    model.z: np.random.normal(
                        0,
                        1,
                        (params.batch_size, params.z_dim)
                    ),
                    model.mask: np.ones_like(x),
                    validation: val,
                    avg_d_loss: np.average(d_losses),
                    avg_g_loss: np.average(g_losses)
                })
                summary_writer.add_summary(summary, step)
                summary_writer.flush()
                d_losses = []
                g_losses = []
        plt.plot(all_losses_d, label="D loss")
        plt.plot(all_losses_g, label="G loss")
        plt.legend()
        plt.show()

def main(args):
    start_time = time.time()
    if not os.path.exists(args.model):
        os.mkdir(args.model)

    with open(os.path.join(args.model, 'params.json'), 'w') as f:
        f.write(json.dumps(vars(args)))

    dataset = data.Dataset(args.dataset)
    x = tf.compat.v1.placeholder(tf.float32, shape=(None, args.vocab_size), name='x')
    z = tf.compat.v1.placeholder(tf.float32, shape=(None, args.z_dim), name='z')
    mask = tf.compat.v1.placeholder(
        tf.float32,
        shape=(None, args.vocab_size),
        name='mask'
    )

    model = m.ADM(x, z, mask, args)
    train(model, dataset, args, wasserstein=args.wasserstein)
    print("--- %s seconds ---" % ((time.time() - start_time)/60.0))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='model22',
                        help='path to model output directory')
    parser.add_argument('--dataset', type=str, default='data/ag_news_preprocessed',
                        help='path to the input dataset')
    parser.add_argument('--vocab-size', type=int, default=2000,
                        help='the vocab size')
    parser.add_argument('--g-dim', type=int, default=300,
                        help='size of generator hidden dimension')
    parser.add_argument('--z-dim', type=int, default=16,
                        help='size of the document encoding')
    parser.add_argument('--noise', type=float, default=0.3,
                        help='masking noise percentage')
    parser.add_argument('--learning-rate', type=float, default=0.0001,
                        help='initial learning rate')
    parser.add_argument('--num-steps', type=int, default=6000,
                        help='the number of steps to train for')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='the batch size')
    parser.add_argument('--num-cores', type=int, default=2,
                        help='the number of CPU cores to use')
    parser.add_argument('--log-every', type=int, default=100,
                        help='print loss after this many steps')
    parser.add_argument('--save-every', type=int, default=500,
                        help='print loss after this many steps')
    parser.add_argument('--validate-every', type=int, default=6000,
                        help='do validation after this many steps')
    parser.add_argument('--wasserstein', type=bool, default=False,
                        help='whether to use wasserstein loss')
    parser.add_argument('--k', type=float, default=0.,
                        help="Initial coefficient")
    parser.add_argument('--gamma', type=float, default=0.5,
                        help="Gamma equilibrium ratio")
    parser.add_argument('--lambd', type=float, default=1e-3,
                        help="Equilibrium control")
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())

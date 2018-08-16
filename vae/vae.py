#!/usr/bin/env python
import os
import numpy as np
import cv2

import tensorflow as tf
from tensorflow.contrib.slim import fully_connected as fc

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

from vm_config import set_train_args
from data_loader import DataLoader

args = set_train_args()
# num_sample = 6000
input_dim = args.target_size ** 2 * 1  # patch image's size^2 * n_channels


class VariantionalAutoencoder(object):
    """
    An encoder-decoder for context images or cropped patches. The image is encoded as z vector
    and is decoded by the decoder to reconstruct the original image. The z vector that could be
    obtained from the pre-trained VAE could be used to combine with the trajectory information
    x, y for further RNN learning in trajectory prediction.

    x --> f0->f1->f2->f3-> fc->    z_mean    \ -> z -> g1->g2->g3->g4-> fc (sigmoid) --> x_hat
                           fc -> z_log_sigma /

    """

    def __init__(self, learning_rate=1e-3, batch_size=100, n_z=5):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_z = n_z      # the size of z vector

        self.x = None
        self.x_hat = None
        self.z = None
        self.z_mu, self.z_log_sigma_sq = None, None
        self.train_op = None
        self.recon_loss, self.latent_loss, self.total_loss = None, None, None

        # Build vae and the loss functions
        self.build()

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

    def build(self):
        self.x = tf.placeholder(name='x', dtype=tf.float32, shape=[None, input_dim])

        # Encode
        # x -> z_mean, z_sigma -> z
        f0 = fc(self.x, 2048, scope='enc_fc0', activation_fn=tf.nn.elu)
        f1 = fc(f0, 512, scope='enc_fc1', activation_fn=tf.nn.elu)
        f2 = fc(f1, 256, scope='enc_fc2', activation_fn=tf.nn.elu)
        f3 = fc(f2, 128, scope='enc_fc3', activation_fn=tf.nn.elu)
        self.z_mu = fc(f3, self.n_z, scope='enc_fc4_mu', activation_fn=None)
        self.z_log_sigma_sq = fc(f3, self.n_z, scope='enc_fc4_sigma', activation_fn=None)
        eps = tf.random_normal(shape=tf.shape(self.z_log_sigma_sq),
                               mean=0, stddev=1, dtype=tf.float32)
        self.z = self.z_mu + tf.sqrt(tf.exp(self.z_log_sigma_sq)) * eps

        # Decode
        # z -> x_hat
        g1 = fc(self.z, 128, scope='dec_fc1', activation_fn=tf.nn.elu)
        g2 = fc(g1, 256, scope='dec_fc2', activation_fn=tf.nn.elu)
        g3 = fc(g2, 512, scope='dec_fc3', activation_fn=tf.nn.elu)
        g4 = fc(g3, 2048, scope='dec_fc4', activation_fn=tf.nn.elu)  ## add
        self.x_hat = fc(g4, input_dim, scope='dec_fc5', activation_fn=tf.sigmoid)

        # Loss
        # Reconstruction loss
        # Minimize the cross-entropy loss
        # H(x, x_hat) = -\Sigma x*log(x_hat) + (1-x)*log(1-x_hat)
        epsilon = 1e-10
        recon_loss = -tf.reduce_sum(
            self.x * tf.log(epsilon+self.x_hat) + (1-self.x) * tf.log(epsilon+1-self.x_hat),
            axis=1
        )
        self.recon_loss = tf.reduce_mean(recon_loss)

        # Latent loss
        # Kullback Leibler divergence: measure the difference between two distributions
        # Here we measure the divergence between the latent distribution and N(0, 1)
        latent_loss = -0.5 * tf.reduce_sum(
            1 + self.z_log_sigma_sq - tf.square(self.z_mu) - tf.exp(self.z_log_sigma_sq), axis=1)
        self.latent_loss = tf.reduce_mean(latent_loss)

        self.total_loss = tf.reduce_mean(recon_loss + latent_loss)

        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.total_loss)

        return

    # Execute the forward and the backward pass
    def run_single_step(self, x, epoch, iter, saver):

        _, x_hat, loss, recon_loss, latent_loss = self.sess.run(
            [self.train_op, self.x_hat, self.total_loss, self.recon_loss, self.latent_loss],
            feed_dict={self.x: x})

        # checkpoint model variable
        if (epoch + 1) % 100 == 0 and iter == num_sample / self.batch_size - 1:
            model_name = 'epoch{}: {}.ckpt'.format(epoch + 1, loss)
            dump_model_full_path = os.path.join('../vae_model/', model_name)
            saver.save(sess=self.sess, save_path=dump_model_full_path)

        return x_hat, loss, recon_loss, latent_loss

    # x -> x_hat
    def reconstructor(self, x, optimal_model_path=None):
        # restore trained vae model
        if optimal_model_path:
            saver = tf.train.Saver()
            saver.restore(self.sess, optimal_model_path)

        # reconstruct
        x_hat = self.sess.run(self.x_hat, feed_dict={self.x: x})
        return x_hat

    # z -> x
    def generator(self, z):
        x_hat = self.sess.run(self.x_hat, feed_dict={self.z: z})
        return x_hat

    # x -> z
    def transformer(self, x):
        z = self.sess.run(self.z, feed_dict={self.x: x})
        return z


# prepare whole context image training data
def next_batch(img_counter, batch_size=100):

    batch = []
    for i in range(batch_size):
        i += 1
        full_path = os.path.join('../dataset/context/', str(img_counter + i) + '_context.jpg')
        context_data = plt.imread(full_path)

        context_data = cv2.resize(context_data, dsize=(args.target_size, args.target_size)) / 500.

        batch.append(context_data.reshape((-1)))  # N, target_size * target_size
    return np.array(batch)


def trainer(learning_rate=1e-3, batch_size=100, num_epoch=1000, n_z=5):

    model = VariantionalAutoencoder(learning_rate=learning_rate,
                                    batch_size=batch_size, n_z=n_z)
    args = set_train_args()

    dataloader = DataLoader(
        data_root_dir=args.data_root_dir,
        train_batch_size=batch_size,
        val_batch_size=batch_size,
        seq_length=1,
        frequency=args.frequency,
        discrete_timestamps=args.discrete_timestamps,
        target_size=args.target_size,
        patch_size=args.patch_size,
        context_files_root_dir=args.context_files_root_dir,
        csv_files_root_dir=args.csv_files_root_dir
    )

    saver = tf.train.Saver(max_to_keep=None)

    for epoch in range(num_epoch):

        img_counter = 0

        for iter in range(dataloader.train_batch_num):

            # prepare a batch of context patch to train, shape = (batch_size, target_size * target_size)
            __, __, batch, __ = dataloader.next_batch("train")

            # execute the forward and the backward pass and report computed losses
            x_hat, loss, recon_loss, latent_loss = model.run_single_step(batch, epoch, iter, saver)
            img_counter += batch_size

        if epoch % 10 == 0:
            print('[Epoch {}] Loss: {}, Recon loss: {}, Latent loss: {}'.format(
                epoch, loss, recon_loss, latent_loss))

    print('Done!')
    return model


if __name__ == "__main__":
    batch_size = 25

    dataloader = DataLoader(
        data_root_dir=args.data_root_dir,
        train_batch_size=batch_size,
        val_batch_size=batch_size,
        seq_length=args.seq_length,
        frequency=args.frequency,
        discrete_timestamps=args.discrete_timestamps,
        target_size=args.target_size,
        patch_size=args.patch_size,
        context_files_root_dir=args.context_files_root_dir,
        csv_files_root_dir=args.csv_files_root_dir
    )

    h = w = args.target_size  # width and height of the original image

    # Train the model
    vae = trainer(learning_rate=1e-4, batch_size=batch_size, num_epoch=500, n_z=args.n_z)
    #vae = VariantionalAutoencoder(learning_rate=1e-4, batch_size=batch_size, n_z=5)

    # Test the trained model: reconstruction
    batch = next_batch(img_counter=0, batch_size=batch_size)
    __, __, batch, __ = dataloader.next_batch("train")

    batch = batch[:batch_size]
    x_reconstructed = vae.reconstructor(batch, optimal_model_path='../vae_model/epoch500: 8270.86425781.ckpt')

    # Visualize reconstructed image against original
    n = np.sqrt(vae.batch_size).astype(np.int32)
    I_reconstructed = np.empty((h*n, 2*w*n))
    for i in range(n):
        for j in range(n):
            x = np.concatenate(
                (x_reconstructed[i*n+j, :].reshape(h, w),
                 batch[i*n+j, :].reshape(h, w)),
                axis=1
            )
            I_reconstructed[i*h:(i+1)*h, j*2*w:(j+1)*2*w] = x

    fig = plt.figure()
    plt.imshow(I_reconstructed, cmap='gray')
    plt.savefig('I_reconstructed1.png')
    plt.close(fig)

import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim import fully_connected as fc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

import cv2

from vm_config import set_train_args
from data_loader import DataLoader

args = set_train_args()
num_sample = 6000
input_dim = args.target_size ** 2 * 1
w = h = args.target_size

class VariantionalAutoencoder(object):
    """
    an encoder-decoder for context image. the z code of the encoder is used to
    combine with the trajectory information for further RNN training.
    """
    
    def __init__(self, learning_rate=1e-3, batch_size=100, n_z=5):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_z = n_z

        self.build()

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

    # Build the netowrk and the loss functions
    def build(self):
        self.x = tf.placeholder(name='x', dtype=tf.float32, shape=[None, input_dim])

        # Encode
        # x -> z_mean, z_sigma -> z
        f0 = fc(self.x, 2048, scope='enc_fc0', activation_fn=tf.nn.elu)  # add
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
        #self.x_hat = tf.clip_by_value(self.x_hat,1e-10,1.0)

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
        self.train_op = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.total_loss)
        return

    # Execute the forward and the backward pass
    def run_single_step(self, x, epoch, iter, saver):
        _, x_hat, loss, recon_loss, latent_loss = self.sess.run(
            [self.train_op, self.x_hat, self.total_loss, self.recon_loss, self.latent_loss],
            feed_dict={self.x: x}
        )
        # checkpoint model variable
        if (epoch + 1) % 100 == 0 and iter == num_sample / self.batch_size - 1:
            model_name = 'epoch{}: {}.ckpt'.format(epoch + 1, loss)
            dump_model_full_path = os.path.join(args.dump_model_para_root_dir, model_name)
            saver.save(sess=self.sess, save_path=dump_model_full_path)

        return x_hat, loss, recon_loss, latent_loss

    # x -> x_hat
    def reconstructor(self, x):
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


def next_batch(img_counter, batch_size=100):
    batch = []
    for i in range(batch_size):
        i += 1

        full_path = os.path.join('../dataset/context/', str(img_counter + i) + '_context.jpg')
        context_data = plt.imread(full_path)
        context_data = cv2.resize(context_data, dsize=(args.target_size, args.target_size)) / 500.
        # print context_data[80][70:90]
        batch.append(context_data.reshape((-1)))  # N, 320 * 320

    return np.array(batch)

def trainer(learning_rate=1e-3, batch_size=100, num_epoch=1000, n_z=5):
    model = VariantionalAutoencoder(learning_rate=learning_rate,
                                    batch_size=batch_size, n_z=n_z)
    saver = tf.train.Saver(max_to_keep=None)

    for epoch in range(num_epoch):
        img_counter = 0

        for iter in range(num_sample // batch_size):

            batch = next_batch(img_counter, batch_size)

            # Execute the forward and the backward pass and report computed losses
            x_hat, loss, recon_loss, latent_loss = model.run_single_step(batch, epoch, iter, saver)
            img_counter += batch_size

        if epoch % 20 == 0:
            print('[Epoch {}] Loss: {}, Recon loss: {}, Latent loss: {}'.format(
                epoch, loss, recon_loss, latent_loss))


    print('Done!')
    return model


if __name__ == "__main__":
    # Train the model
    args = set_train_args()
    model = trainer(learning_rate=1e-4,  batch_size=100, num_epoch=500, n_z=args.n_z)
    model = VariantionalAutoencoder(learning_rate=1e-4,
                                    batch_size=100, n_z=5)
    # /home/yuhuang/Desktop/WorldModelPredictor/vae_model/epoch200: 8334.05664062.ckpt.data-00000-of-00001
    # model_loading_path_vae = '../vae_model/epoch200: 8334.05664062.ckpt'
    # saver = tf.train.Saver()
    # saver.restore(tf.Session(), model_loading_path_vae)

    # Test the trained model: reconstruction
    batch = next_batch(img_counter=0, batch_size=100)
    x_reconstructed = model.reconstructor(batch)

    n = np.sqrt(model.batch_size).astype(np.int32)
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
    plt.savefig('I_reconstructed0.png')
    plt.close(fig)

    # # Test the trained model: generation
    # # Sample noise vectors from N(0, 1)
    # z = np.random.normal(size=[model.batch_size, model.n_z])
    # x_generated = model.generator(z)
    #
    # n = np.sqrt(model.batch_size).astype(np.int32)
    # I_generated = np.empty((h*n, w*n))
    # for i in range(n):
    #     for j in range(n):
    #         I_generated[i*h:(i+1)*h, j*w:(j+1)*w] = x_generated[i*n+j, :].reshape(28, 28)
    #
    # fig = plt.figure()
    # plt.imshow(I_generated, cmap='gray')
    # plt.savefig('I_generated.png')
    # plt.close(fig)
    #
    # tf.reset_default_graph()
    # # Train the model with 2d latent space
    # model_2d = trainer(learning_rate=1e-4,  batch_size=100, num_epoch=50, n_z=2)
    #
    # # Test the trained model: transformation
    # batch = mnist.test.next_batch(3000)
    # z = model_2d.transformer(batch[0])
    # fig = plt.figure()
    # plt.scatter(z[:, 0], z[:, 1], c=np.argmax(batch[1], 1))
    # plt.colorbar()
    # plt.grid()
    # plt.savefig('I_transformed.png')
    # plt.close(fig)
    #
    # # Test the trained model: transformation
    # n = 20
    # x = np.linspace(-2, 2, n)
    # y = np.linspace(-2, 2, n)
    #
    # I_latent = np.empty((h*n, w*n))
    # for i, yi in enumerate(x):
    #     for j, xi in enumerate(y):
    #         z = np.array([[xi, yi]]*model_2d.batch_size)
    #         x_hat = model_2d.generator(z)
    #         I_latent[(n-i-1)*28:(n-i)*28, j*28:(j+1)*28] = x_hat[0].reshape(28, 28)
    #
    # fig = plt.figure()
    # plt.imshow(I_latent, cmap="gray")
    # plt.savefig('I_latent.png')
    # plt.close(fig)

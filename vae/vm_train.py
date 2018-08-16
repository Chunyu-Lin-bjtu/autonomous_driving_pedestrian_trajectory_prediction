
import os
import numpy as np

import tensorflow as tf

from data_loader import DataLoader
from vm_model import VMModel
from vae import VariantionalAutoencoder
from vm_config import set_train_args

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

import cv2
import shutil

def train():
    """
    Train phase main process

    :return:
    """
    args = set_train_args()

    # remove previous summary logs
    for folder in (args.train_summary, args.val_summary):
        if os.path.exists(folder):
            shutil.rmtree(folder)

    dataloader = DataLoader(
        data_root_dir=args.data_root_dir,
        train_batch_size=args.batch_size,
        val_batch_size=args.batch_size,
        seq_length=args.seq_length,
        frequency=args.frequency,
        discrete_timestamps=args.discrete_timestamps,
        target_size=args.target_size,
        patch_size=args.patch_size,
        context_files_root_dir=args.context_files_root_dir,
        csv_files_root_dir=args.csv_files_root_dir
    )

    # load pre-trained vae model. vae should have been finished training before executing this script
    vae = VariantionalAutoencoder(learning_rate=1e-4, batch_size=100, n_z=args.n_z)
    saver = tf.train.Saver()
    saver.restore(tf.Session(), args.model_loading_path_vae)
    freeze_variables = tf.global_variables()

    print '\nDatasets loaded! \nReady to train.'

    # build model
    vmmodel = VMModel(args)

    print '\n\nModel initialized successfully.'

    # to save log
    train_error = np.zeros(args.num_epochs)
    valid_error = np.zeros(args.num_epochs)

    step = 0

    # configure GPU training, soft allocation.
    gpuConfig = tf.ConfigProto(allow_soft_placement=True)
    gpuConfig.gpu_options.allow_growth = True

    with tf.Session(config=gpuConfig) as sess:

            train_writer = tf.summary.FileWriter('./train/', sess.graph)
            val_writer = tf.summary.FileWriter('./val/', sess.graph)

            # initialize all variables in the graph
            #tf.global_variables_initializer().run()

            # initialize vm model parameters while leave vae model untouched
            uninitialized_variables = set(sess.run(tf.report_uninitialized_variables()))
            tf.variables_initializer(
                [v for v in tf.global_variables() if v.name.split(':')[0] not in freeze_variables]).run()

            # initialize a saver that saves all the variables in the graph
            saver = tf.train.Saver(max_to_keep=None)

            print '\n\nStart Training... \n'
            for e in range(args.num_epochs):

                dataloader.reset_train_pointer()
                dataloader.reset_val_pointer()

                # --- TRAIN ---
                for batch in range(dataloader.train_batch_num):

                    # context data shape: (N * seq_length, target_size * target_size)
                    # input batch shape: (N, seq_length, feature_size)
                    # target batch shape: (N, seq_length, discrete_steps, feature_size)
                    # The context images here have been resized down to target_size, and values are divided by 500
                    input_data, target_data, context_patches, context = dataloader.next_batch(mode='train')

                    # encode context by vae to obtain the z code
                    z = vae.transformer(context_patches)
                    context = cv2.resize(context[0], (160, 160)) / 500.
                    x_hat = vae.reconstructor([context.reshape(-1)])
                    print x_hat.shape
                    plt.imshow(context)
                    plt.show()
                    plt.imshow(x_hat[0].reshape(160, 160))
                    plt.show()
                    z = z.reshape((-1, args.seq_length, args.n_z ))

                    complete_input, batch_error_train, summary, __ = sess.run(
                        fetches=[
                            vmmodel.complete_input,
                            vmmodel.loss,
                            vmmodel.summary_op,
                            vmmodel.train_op,
                        ],
                        feed_dict={
                            vmmodel.input_data: input_data,
                            vmmodel.target_data: target_data,
                            vmmodel.z: z
                        })
                    # add summary and accumulate stats
                    train_writer.add_summary(summary, step)
                    train_error[e] += batch_error_train
                    step += 1

                # normalise running means by number of batches
                train_error[e] /= dataloader.train_batch_num

                # --- VALIDATION ---
                for batch in range(dataloader.val_batch_num):
                    input_data, target_data, context_patches, __ = dataloader.next_batch(mode='val')
                    z = vae.transformer(context_patches)
                    z = z.reshape((-1, args.seq_length, args.n_z))

                    summary, predicted_outputs, batch_error_val = sess.run(
                        fetches=[vmmodel.summary_op, vmmodel.predicted_outputs, vmmodel.loss],
                        feed_dict={vmmodel.input_data: input_data, vmmodel.target_data: target_data, vmmodel.z: z})
                    val_writer.add_summary(summary, step)
                    valid_error[e] += batch_error_val
                valid_error[e] /= dataloader.val_batch_num

                # checkpoint model variable
                if (e + 1) % args.save_every_epoch == 0:
                    model_name = 'epoch{}_{:2f}' \
                                 '_{:2f}.ckpt'.format(e + 1, train_error[e], valid_error[e])

                    dump_model_full_path = os.path.join(args.dump_model_para_root_dir, model_name)
                    # save model
                    saver.save(sess=sess, save_path=dump_model_full_path)
                    tf.add_to_collection("predict", vmmodel.predicted_outputs)

                print('Epoch {0:02d}: err(train)={1:.2f}, err(valid)={2:.2f}' .format(e + 1, train_error[e], valid_error[e]))

    # close writer and session objects
    train_writer.close()
    val_writer.close()
    sess.close()

    return train_error, valid_error


if __name__ == '__main__':

    train_error, valid_error = train()

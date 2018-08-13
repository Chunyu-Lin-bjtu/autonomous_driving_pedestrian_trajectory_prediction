import os
import tensorflow as tf
import numpy as np

import shutil

from data_loader import DataLoader
from vru2_model import VRU2Model
from vru_config import set_train_args


def train(data_root_dir, context_files_root_dir, csv_files_root_dir, dump_model_para_root_dir):
    """
    Train phase main process

    :return:
    """
    args = set_train_args()

    # remove older summary log and model parameters
    for folder in (args.train_summary, args.val_summary, dump_model_para_root_dir):
        if os.path.exists(folder):
            shutil.rmtree(folder)

    dataloader = DataLoader(
        data_root_dir=data_root_dir,
        train_batch_size=args.batch_size,
        val_batch_size=args.batch_size,
        seq_length=args.seq_length,
        frequency=args.frequency,
        discrete_timestamps=args.discrete_timestamps,
        patch_size=args.patch_size,
        context_files_root_dir=context_files_root_dir,
        csv_files_root_dir=csv_files_root_dir
    )

    print '\nDatasets loaded! \nReady to train.'

    # create model
    vrumodel = VRU2Model(args)

    print '\n\nModel initialized successfully.'
    # to save log
    train_error = np.zeros(args.num_epochs)
    valid_error = np.zeros(args.num_epochs)

    step = 0

    # configure GPU training, soft allocation.
    gpuConfig = tf.ConfigProto(allow_soft_placement=True)
    gpuConfig.gpu_options.allow_growth = True

    with tf.Session(config=gpuConfig) as sess:

            train_writer = tf.summary.FileWriter(args.train_summary, sess.graph)
            val_writer = tf.summary.FileWriter(args.val_summary, sess.graph)

            # initialize all variables in the graph
            tf.global_variables_initializer().run()

            # initialize a saver that saves all the variables in the graph
            saver = tf.train.Saver(max_to_keep=None)

            print '\n\nStart Training... \n'
            for e in range(args.num_epochs):

                dataloader.reset_train_pointer()
                dataloader.reset_val_pointer()

                # --- TRAIN ---
                for batch in range(dataloader.train_batch_num):

                    # context data shape: (batch_size*seq_length, patch_size**2)
                    # input batch shape: (batch_size, seq_length, feature_size)
                    # target batch shape: (batch_size, discrete_steps, feature_size)
                    input_batch, target_batch, context_patch, __ = dataloader.next_batch(mode='train')

                    # reshape to  4D tensor [batch_size*seq_length, patch_size, patch_size, in_channel]
                    context_data_feed = context_patch.reshape((-1, args.patch_size, args.patch_size, 1))

                    complete_input, batch_error_train, summary, __ = sess.run(
                        fetches=[
                            vrumodel.complete_input,
                            vrumodel.loss,
                            vrumodel.summary_op,
                            vrumodel.train_op,
                        ],
                        feed_dict={
                            vrumodel.input_data: input_batch,
                            vrumodel.target_data: target_batch,
                            vrumodel.context_input: context_data_feed
                        })
                    # add summary and accumulate stats
                    train_writer.add_summary(summary, step)
                    train_error[e] += batch_error_train
                    step += 1

                # normalise running means by number of batches
                train_error[e] /= dataloader.train_batch_num

                # --- VALIDATION ---
                for batch in range(dataloader.val_batch_num):
                    input_batch, target_batch, context_patch, __ = dataloader.next_batch(mode='val')

                    context_patch = context_patch.reshape((-1, args.patch_size, args.patch_size, 1))

                    summary, predicted_outputs, batch_error_val = sess.run(
                        fetches=[vrumodel.summary_op, vrumodel.predicted_outputs, vrumodel.loss],
                        feed_dict={
                            vrumodel.input_data: input_batch,
                            vrumodel.target_data: target_batch,
                            vrumodel.context_input: context_patch
                        })
                    val_writer.add_summary(summary, step)
                    valid_error[e] += batch_error_val
                valid_error[e] /= dataloader.val_batch_num

                # checkpoint model variable
                if (e + 1) % args.save_every_epoch == 0:
                    model_name = 'epoch{}_{:2f}' \
                                 '_{:2f}.ckpt'.format(e + 1, train_error[e], valid_error[e])

                    dump_model_full_path = os.path.join(dump_model_para_root_dir, model_name)
                    # save model
                    saver.save(sess=sess, save_path=dump_model_full_path)
                    tf.add_to_collection("predict", vrumodel.predicted_outputs)

                print('Epoch {0:02d}: err(train)={1:.2f}, err(valid)={2:.2f}' .format(e + 1, train_error[e], valid_error[e]))

    # close writer and session objects
    train_writer.close()
    val_writer.close()
    sess.close()

    return train_error, valid_error


if __name__ == '__main__':

    train_error, valid_error = train(data_root_dir='../../dataset_data/',
                context_files_root_dir='../../dataset/context/',
                csv_files_root_dir='../../dataset/trajectory/',
                dump_model_para_root_dir='../vru2_model/')

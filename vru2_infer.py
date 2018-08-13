#!/usr/bin/env python
from vru_config import *
from vru2_model import *
from data_loader import *

import matplotlib.pyplot as plt

import random
import argparse


def set_infer_args():
    parser = argparse.ArgumentParser()

    # File dump path for visualization jpg
    parser.add_argument('--infer_dump_path', type=str, default='',
                        help='dump path of visualization pic')

    # path of the optimal model
    parser.add_argument('--optimal_model_path', type=str, default='epoch156_201.193525_201.908265.ckpt',
                        help='optimal model path to load from')

    # number of samples to visusalize
    parser.add_argument('--num_samples', type=int, default=5,
                        help='number of random samples in batch to visualize')

    args = parser.parse_args()
    return args


def visualize_prediction(past, ground_truth, predicted, context, dump_path):
    # plot past, predicted and ground truth trajectory on the context on the same graph
    plt.imshow(context)

    for traj in [past, ground_truth, predicted]:
        plt.scatter(traj[:, 1], traj[:, 0])

    plt.show()
    plt.savefig(dump_path)


if __name__ == '__main__':
    args = set_train_args()
    args_infer = set_infer_args()
    tf.reset_default_graph()

    # configure GPU training, soft allocation.
    gpuConfig = tf.ConfigProto(allow_soft_placement=True)
    gpuConfig.gpu_options.allow_growth = True
    with tf.Session(config=gpuConfig) as sess:

        data_root_dir = '../../dataset_data/'
        context_files_root_dir = '../../dataset/context/'
        csv_files_root_dir = '../../dataset/trajectory/'
        dump_model_para_root_dir = '../vru2_model/'
        model_loading_path = os.path.join(dump_model_para_root_dir, args_infer.optimal_model_path)

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

        print "\n\nfinished loading datasets\n\n"

        vrumodel = VRU2Model(args)

        # load trained model
        saver = tf.train.Saver()
        saver.restore(sess, model_loading_path)

        print "model loaded.\n\n"

        for n in range(1):

            # context data shape: (batch_size*seq_length, patch_size**2)
            # input batch shape: (batch_size, seq_length, feature_size)
            # target batch shape: (batch_size, discrete_steps, feature_size)
            # context data whole shape: (batch_size, width, height)
            input_batch, target_batch, context_patch, context_data_whole = dataloader.next_batch(mode='val')
            print "obtained infer batch. \n\n"

            context_data_feed = context_patch.reshape((args.batch_size*args.seq_length, args.patch_size, args.patch_size, 1))

            print "starts predicting.\n\n"

            # Get prediction results
            # tf_predicted_outputs transformed predicted_outputs to the absolute coordinates on context,
            # tf_predicted_outputs shape = (N, seq_length , discrete_steps, feature_size)
            complete_input, predicted_outputs, loss, tf_predicted_outputs, target_delta = sess.run(
                fetches=[
                    vrumodel.complete_input,
                    vrumodel.predicted_outputs,
                    vrumodel.loss,
                    vrumodel.tf_predicted_outputs,
                    vrumodel.target_data_delta
                ],
                feed_dict={
                    vrumodel.input_data: input_batch,
                    vrumodel.target_data: target_batch,
                    vrumodel.context_input: context_data_feed
                })

            # pick a random sample from batch to visualize
            pick_serial = random.randint(0, args.batch_size-1)
            sample_name = 'vru_result_'+str(n)+'.png'
            sample_dump_path = os.path.join(args_infer.infer_dump_path, sample_name)

            print 'visualizing predicting results.\n\n'

            visualize_prediction(input_batch[pick_serial],
                                 target_batch[pick_serial],
                                 tf_predicted_outputs[pick_serial][-1],  # shape=(discrete_steps, feature_size)
                                 context_data_whole[pick_serial],
                                 sample_dump_path)

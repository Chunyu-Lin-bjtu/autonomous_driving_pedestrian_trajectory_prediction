from vru_config import *
from vru_model import *
from data_loader import *

import matplotlib.pyplot as plt

import random
import argparse

# This script does the following:
# load from trained models to visualize on predicted trajectories and
# compared with the ground truth in context

def set_infer_args():
    parser = argparse.ArgumentParser()

    # File dump path for visualization jpg
    parser.add_argument('--infer_dump_path', type=str, default='',
                        help='dump path of visualization pic')

    # Optimal model path to load from
    parser.add_argument('--optimal_model_path', type=str, default='epoch10_246.721438_240.188778.ckpt',
                        help='optimal model path to load from')

    # number of samples to visusalize
    parser.add_argument('--num_samples', type=int, default=5,
                        help='number of random samples in batch to visualize')

    args = parser.parse_args()
    return args


def visualize_prediction(past, ground_truth, predicted, context, dump_path):
    """
    Visualize on jpg the past trajectory, future trajectory and predicted future
    trajectory in context.

    :param past:
    :param ground_truth:
    :param predicted:
    :param context:
    :param dump_path:
    """
    plt.imshow(context)
    for traj in [past, ground_truth, predicted]:
        plt.scatter(traj[:, 1], traj[:, 0])
    plt.show()
    plt.savefig(dump_path)


if __name__ == '__main__':
    args = set_train_args()
    args_infer = set_infer_args()
    tf.reset_default_graph()

    with tf.Session() as sess:
        data_root_dir = '../../dataset_data/'
        context_files_root_dir = '../../dataset/context/'
        csv_files_root_dir = '../../dataset/trajectory/'
        dump_model_para_root_dir = '../vru_model/'
        model_loading_path = os.path.join(dump_model_para_root_dir, args_infer.optimal_model_path)

        # prepare test data
        dataloader = DataLoader(
            data_root_dir=data_root_dir,  # '../../dataset_data/'
            train_batch_size=args.batch_size,
            val_batch_size=args.batch_size,
            seq_length=args.seq_length,
            frequency=args.frequency,
            discrete_timestamps=args.discrete_timestamps,
            context_files_root_dir=context_files_root_dir,
            csv_files_root_dir=csv_files_root_dir
        )

        # for i in range(10):
        #     dataloader.next_batch(mode='val')
        (context_data, input_batch), target_batch = dataloader.next_batch(mode='val')
        context_data_feed = context_data.reshape((-1, args.width, args.height, 1))

        # initialize model
        vrumodel = VRUModel(args)

        # load trained model
        saver = tf.train.Saver()
        saver.restore(sess, model_loading_path)

        # Get prediction results
        predicted_outputs, loss, outputs = sess.run(
            fetches=[
                vrumodel.predicted_outputs,
                vrumodel.loss,
                vrumodel.tf_predicted_outputs
            ],
            feed_dict={
                vrumodel.input_data: input_batch,
                vrumodel.target_data: target_batch,
                vrumodel.context_input: context_data_feed
            })

        for n in range(1):

            # pick a random sample from batch to visualize
            pick_serial = random.randint(0, args.batch_size-1)
            sample_name = 'vru_result_'+str(n)+'.png'
            sample_dump_path = os.path.join(args_infer.infer_dump_path, sample_name)

            context_data = context_data.reshape((args.batch_size, args.height, args.width))

            visualize_prediction(input_batch[pick_serial],
                                 target_batch[pick_serial],
                                 outputs[pick_serial],
                                 context_data[pick_serial],
                                 sample_dump_path)

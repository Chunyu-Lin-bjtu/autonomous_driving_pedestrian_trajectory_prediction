
import argparse

def set_train_args():
    """
    configure the training arguments

    :return: args
    """
    parser = argparse.ArgumentParser()

    # size of each batch for training dataset parameter
    parser.add_argument('--batch_size', type=int, default=32,
                        help='training set minibatch size')

    # size of history window not including the current point
    parser.add_argument('--history_window', type=int, default=32,
                        help='size of history window trajectory')
    # predict frequency
    parser.add_argument('--frequency', type=int, default=20,
                        help='predict frequency')

    parser.add_argument('--seq_length', type=int, default=64,
                        help='length of the time sequence')

    parser.add_argument('--feature_size', type=int, default=2,
                        help='feature size')

    # number of discrete steps
    parser.add_argument('--discrete_timestamps', type=list, default=[1, 2, 3, 4, 5, 6],
                        help='List of the specific timestamp to be predict')

    # patch size / mask size
    parser.add_argument('--patch_size', type=int, default=1280,
                        help='input patch size')

    # size of resized patch
    parser.add_argument('--target_size', type=int, default=160,
                        help='size of resized patch, ready to feed in network')

    # learning rate for unet parameter
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='learning rate for RNN')

    # z value size in vae
    parser.add_argument('--n_z', type=int, default=100,         # 5
                        help='vae network z value size')

    parser.add_argument('--rnn_size', type=int, default=126,
                        help='the output size of rnn')

    parser.add_argument('--n_layers', type=int, default=2,
                        help='the output size of rnn')

    # Dropout probability parameter
    parser.add_argument('--keep_prob', type=float, default=0.8,
                        help='dropout keep probability')

    # -- training
    # Number of epochs parameter
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='number of epochs')

    parser.add_argument('--save_every_epoch', type=int, default=1,
                        help='save model variables for every how many epoch(s)')

    # directory for train summary FileWriter
    parser.add_argument('--train_summary', type=str, default='./train/',
                        help='train summary FileWriter')

    # directory for val summary FileWriter
    parser.add_argument('--val_summary', type=str, default='./val/',
                        help='val summary FileWriter')

    # mode
    parser.add_argument('--mode', type=str, default='train',
                        help='train or infer')

    # root directory of dataset data
    parser.add_argument('--data_root_dir', type=str, default='../dataset_data/',
                        help='root directory of dataset data')

    # root directory of context jpg image files
    parser.add_argument('--context_files_root_dir', type=str, default='../dataset/context/',
                        help='root directory of context jpg image files')

    # root directory of trajectory csv files
    parser.add_argument('--csv_files_root_dir', type=str, default='../dataset/trajectory/',
                        help='root directory of trajectory csv files')

    # directory path to dump model parameters while training
    parser.add_argument('--dump_model_para_root_dir', type=str, default='../vm_model/',
                        help='directory path to dump model parameters while training')

    parser.add_argument('--model_loading_path_vae', type=str, default='../vae_model/epoch500: 8270.86425781.ckpt',
                        help='pre-trained vae model parameters loading path')

    # ---infer
    # File dump path for visualization jpg
    parser.add_argument('--infer_dump_path', type=str, default='',
                        help='dump path of visualization pic')

    # /home/yuhuang/Desktop/WorldModelPredictor/vm_model/epoch53_37.866946_35.920916.ckpt.meta
    parser.add_argument('--optimal_model_path', type=str, default='../vm_model/epoch53_37.866946_35.920916.ckpt',
                        help='optimal model path to load from')

    # number of samples to visusalize
    parser.add_argument('--num_samples', type=int, default=5,
                        help='number of random samples in batch to visualize')


    args = parser.parse_args()
    return args


import argparse

def set_train_args():
    """
    configure the training arguments

    :return: args
    """
    parser = argparse.ArgumentParser()

    # -- model
    # size of each batch for training dataset parameter
    parser.add_argument('--batch_size', type=int, default=32,
                        help='training set minibatch size')

    # size of history window not including the current point
    parser.add_argument('--history_window', type=int, default=32,
                        help='size of history window trajectory')
    # predict frequency
    parser.add_argument('--frequency', type=int, default=20,
                        help='predict frequency')

    # sequence length parameter
    parser.add_argument('--seq_length', type=int, default=64,
                        help='length of the time sequence')

    # feature size parameter
    parser.add_argument('--feature_size', type=int, default=2,
                        help='feature size')

    # number of discrete steps
    parser.add_argument('--discrete_timestamps', type=list, default=[1, 2, 3, 4, 5, 6],
                        help='List of the specific timestamp to be predict')

    # size of the cropped patch from the whole context parameter
    parser.add_argument('--patch_size', type=int, default=640,
                        help='size of cropped patch')

    # target size of resized patch or image parameter, also the input dim of rnn
    parser.add_argument('--target_size', type=int, default=160,
                        help='size of resized patch, ready to feed in network')

    # learning rate for unet parameter
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='learning rate')

    # z value size parameter in vae
    parser.add_argument('--n_z', type=int, default=100,
                        help='size of the z vector in vae')

    # output size of rnn parameter
    parser.add_argument('--rnn_size', type=int, default=126,
                        help='the output size of rnn')

    # number of layers of each rnn cell parameter
    parser.add_argument('--n_layers', type=int, default=2,
                        help='number of layers of rnn unit')

    # Dropout probability parameter
    parser.add_argument('--keep_prob', type=float, default=0.8,
                        help='dropout keep probability')

    # -- training
    # Number of epochs parameter
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='number of epochs')

    # how many epochs to save model parameters during training parameter
    parser.add_argument('--save_every_epoch', type=int, default=5,
                        help='save model variables for every how many epoch(s)')

    # directory for train summary FileWriter
    parser.add_argument('--train_summary', type=str, default='./train/',
                        help='train summary FileWriter')

    # directory for val summary FileWriter
    parser.add_argument('--val_summary', type=str, default='./val/',
                        help='val summary FileWriter')

    # mode
    parser.add_argument('--mode', type=str, default='train',
                        help='mode train or infer')

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
                        help='path to dump model parameters')

    # path of optimal vae model parameter  /home/yuhuang/Desktop/WorldModelPredictor/vm_model/epoch500: 8273.20214844.ckpt.meta
    parser.add_argument('--model_loading_path_vae', type=str, default='../vae_model/epoch89: 7940.46630859.ckpt',
                        help='path of pre-trained vae model parameters')

    # ---infer
    # path to dump visualization images
    parser.add_argument('--infer_dump_path', type=str, default='',
                        help='dump path of visualization pic')

    # path of optimal vm model parameter
    parser.add_argument('--optimal_model_path', type=str, default='../vm_model/epoch53_37.866946_35.920916.ckpt',
                        help='full path of optimal model parameters')

    # number of samples to visusalize in each batch paramter
    parser.add_argument('--num_samples', type=int, default=5,
                        help='number of random samples in batch to visualize')

    args = parser.parse_args()
    return args

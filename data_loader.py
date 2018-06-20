import os
import random

import csv
import pandas as pd
import cPickle as pickle

import numpy as np
import matplotlib.pyplot as plt

class DataLoader(object):
    def __init__(self,
                 data_root_dir,  # '../../dataset/'
                 train_batch_size,  # 32
                 val_batch_size,  # 32
                 seq_length,  # 64
                 frequency,  # 20 Hertz
                 discrete_timestamps,   # [1,2,3,4,5]
                 context_files_root_dir='../../dataset/context/',
                 csv_files_root_dir='../../dataset/trajectory/'
                 ):
        self.context_files_root_dir = context_files_root_dir
        self.csv_files_root_dir = csv_files_root_dir
        self.data_root_dir = data_root_dir
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.seq_length = seq_length
        self.frequency = frequency
        self.discrete_timestamps = discrete_timestamps

        self._train_pointer = 0
        self._val_pointer = 0
        self._train_batch_num = 0
        self._val_batch_num = 0

        # target generation path
        self.data_full_path_train = os.path.join(self.data_root_dir, 'train_set.pkl')
        self.data_full_path_val = os.path.join(self.data_root_dir, 'val_set.pkl')

        # generate data to local if not done yet
        if not os.path.exists(self.data_root_dir):
            os.makedirs(self.data_root_dir)
            self.data_generator(self.data_full_path_train, 'train')     # generate train file
            self.data_generator(self.data_full_path_val, 'val')        # generate val file

    def data_generator(self, data_full_path, mode):
        """
        Generate dataset from context and trajectory csv for training or validation.
        Maps are split into 4000 to make training datasets and 2000 for validation datasets.

        Dataset is context path, input data, target data;
        input data is a list of input trajectory, context data will be read
        from context path only when next_batch is called,
        target data is a list of target trajectory array.

        Each trajectory array is shaped to ([x0, y0], [x1, y1], [x2, y2]...)

        :param data_full_path: csv full path to generate datasets into
        :param mode: train or val
        :return:
        """

        if mode == 'train':
            map_num, s = 4000, 1
        elif mode == 'val':
            map_num, s = 2000, 4001
        else:
            raise RuntimeError("Mode can only be train or val.")

        # map_num = 2 # for test
        dataset_counter = 0

        with open(data_full_path, 'wb') as f:

            for i in range(map_num):

                print '\nGenerating data from map ' + str(i + s) + '_context.jpg ...'

                context_full_path = os.path.join(self.context_files_root_dir, str(i + s) + '_context.jpg')
                csv_full_path = os.path.join(self.csv_files_root_dir, str(i + s) + '_traj.csv')

                # counter to count for number of samples made from one trajectory
                counter = 0

                # read a full trajectory
                df = pd.read_csv(csv_full_path, dtype=float).values

                # length of target trajectory
                target_len = 1 + (len(self.discrete_timestamps) - 1) * self.frequency
                # number of samples available from this trajectory
                num = len(df) - self.seq_length - target_len + 1

                if num < 0:
                    print 'WARNING: ' + str(i + s) + '_traj.csv not long enough to make a full sample data.\nrequired ' \
                          'length: ' + str(self.seq_length + target_len + self.seq_length - 1) + ', given: ' + str(len(df)) + \
                          '\nreduce time step or predict frequency.'
                    continue

                all_feed_samples = []   # hold all data made from one map

                for n in range(num):

                    input_data = df[n:n + self.seq_length]  # shape (64,2)
                    target_data = self.get_weight_points(df, n + self.seq_length, target_len)  # (discrete_step, 2)

                    all_feed_samples.append([context_full_path, input_data, target_data])
                    counter += 1

                print '...' + str(counter) + ' input data have been created.'

                dataset_counter += counter           # increment batch counter

                random.shuffle(all_feed_samples)     # shuffle samples within one map

                for row in all_feed_samples:
                    pickle.dump(row, f)

        # Calculate batch number
        if mode == 'train':
            self._train_batch_num = int(dataset_counter / self.train_batch_size)
        if mode == 'val':
            self._val_batch_num = int(dataset_counter / self.val_batch_size)

    def next_batch(self, mode):
        """
        get the next batch data for training for validation

        :param mode: train or val
        :return: (context data (width*height, ), input trajectory (seq_length, 2)), target trajectory (discrete step, 2)
        """

        if mode == 'train':
            batch_data = []

            f = file(self.data_full_path_train, 'rb')

            # omit used batches
            for idx in range(self._train_pointer):
                pickle.load(f)
            # get next batch
            for i in range(self.train_batch_size):
                batch_data.append(pickle.load(f))

            # get context path, input and target data
            context_matrix, input_data, target_data = self.do_batch_data(batch_data)

            # get context data from path
            context_data = self.feed_context_data(context_matrix)

            # increment sample pointer
            self._train_pointer += self.train_batch_size

            return (context_data, input_data), target_data

        if mode == 'val':

            batch_data = []

            f = file(self.data_full_path_val, 'rb')
            for idx in range(self._val_pointer):
                pickle.load(f)
            for i in range(self.val_batch_size):
                batch_data.append(pickle.load(f))

            context_matrix, input_data, target_data = self.do_batch_data(batch_data)
            context_data = self.feed_context_data(context_matrix)

            self._val_pointer += self.val_batch_size

            return (context_data, input_data), target_data

        else:
            raise RuntimeError('Mode can only be train or val.')

    def get_weight_points(self, df, start_idx, target_len):
        """
        Get the target trajectory with specified frequency

        :param df: full trajectory
        :param start_idx: the point to start with
        :param target_len: the length of target trajectory
        :return: array of weight points
        """
        target = []

        full_target_path = df[start_idx:start_idx + target_len]
        weight_points = [full_target_path[i * self.frequency] for i in range(len(self.discrete_timestamps))]

        for weight_point in weight_points:
            target.append(weight_point)  #weight_point[0]

        return np.array(target)

    def feed_context_data(self, all_context_path):
        """
        Read a list of context path to context data

        :param all_context_path: array of context path
        :return: an array of context data of shape (height * width, )
        """
        all_context_data = []
        print all_context_path
        for p in all_context_path:
            context_data = plt.imread(p)
            all_context_data.append(context_data.reshape((-1,)))

        return np.array(all_context_data)

    def do_batch_data(self, batch_data):
        """
        get context data, inputd data and target data from batch data

        :param batch_data: list of samples
        :return: context matrix, input data, target data
        """
        context_matrix = []
        input_data = []
        target_data = []
        for i in range(len(batch_data)):
            each_sample = batch_data[i]

            context_matrix.append(each_sample[0])
            input_data.append(each_sample[1])
            target_data.append(each_sample[2])

        return context_matrix, input_data, target_data

    def reset_train_pointer(self):
        self._train_pointer = 0

    def reset_val_pointer(self):
        self._val_pointer = 0

    @property
    def train_batch_num(self):
        return self._train_batch_num

    @property
    def val_batch_num(self):
        return self._val_batch_num


if __name__ == '__main__':
    data_loader = DataLoader(data_root_dir='../../dataset_data/',
                             train_batch_size=32,
                             val_batch_size=32,
                             seq_length=64,
                             frequency=20,
                             discrete_timestamps=[1,2,3,4,5,6])
    # (context, input_data), target_data = data_loader.next_batch('val')
    # print input_data[0]
    # (context, input_data), target_data = data_loader.next_batch('val')
    # print input_data[0]
    print data_loader.train_batch_num
    print data_loader.val_batch_num

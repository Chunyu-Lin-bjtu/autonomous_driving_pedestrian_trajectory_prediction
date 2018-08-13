import tensorflow as tf
from tensorflow.contrib import rnn, layers
import numpy as np


class VRU2Model(object):
    """
    Structure of VRU2 prediction model.
                               input trajectory
                                             + => complete_input === RNN ==> predicted trajectory
        context patch == CNN ==> initial_output
    """

    def __init__(self, args):
        """
        configure model parameters
        :param args: model configuration parameters
        :param mode: train or val
        """
        self._batch_size = args.batch_size
        self._feature_size = args.feature_size
        self._mode = args.mode

        # CNN parameters
        self._in_channel = 1
        self._in_width = args.patch_size      # context w
        self._in_height = args.patch_size    # context h
        self._target_size_cnn = args.target_size_cnn
        self._filter_size = args.filter_size
        self._dropout = args.dropout

        # RNN parameters
        self._discrete_steps = len(args.discrete_timestamps)
        self._target_size = self._discrete_steps * args.feature_size # output size of fc layer after rnn

        self._rnn_size = args.rnn_size
        self._seq_length = args.seq_length
        self._n_layers = args.n_layers
        self._keep_prob = args.keep_prob
        self._lr = args.learning_rate

        # optimizer global step
        self._global_step = tf.Variable(0, trainable=False)
        # sequence length argument
        self._sequence_length = tf.convert_to_tensor([self._seq_length for _ in range(self._batch_size)])

        # past trajectory
        self._input_data = tf.placeholder(
            tf.float32, shape=[
                self._batch_size, self._seq_length, self._feature_size], name='input_data')

        # context path to feed into CNN
        self._context_input = tf.placeholder(
            tf.float32, shape=[
            self._batch_size * self._seq_length, self._in_width, self._in_height, self._in_channel],name='context_data')

        # ground true trajectory
        self._target_data = tf.placeholder(
            tf.float32, shape=[
                self._batch_size, self._seq_length, self._discrete_steps, self._feature_size], name='target_data')

        # transformed delta ground truth
        self._target_data_delta = tf.placeholder(
            tf.float32, shape=[
                self._batch_size, self._seq_length, self._discrete_steps, self._feature_size], name='target_data_delta')

        def conv2d(input, num_output, kernel_size, stride):
            # define a cnn unit: x--> conv -> relu -> max_pooling
            # take 4D input tensor
            x = layers.conv2d(inputs=input,
                              num_outputs=num_output,
                              kernel_size=kernel_size,
                              stride=stride,
                              activation_fn=tf.nn.relu)
            x = tf.contrib.layers.max_pool2d(x, kernel_size=[2, 2], stride=1)

            return x

        # ---- CNN ----
        # Convolution to map context data to vector
        # (N * seq_length, patch_size, patch_size, 1) -> 4 conv -> 3fc
        # output = (N * seq_length, target_size_cnn)
        with tf.variable_scope("conv_net"):
            # input, num_channel, kernel_size, stride
            x = conv2d(self._context_input, 4, 11, 7)
            x = conv2d(x, 8, 7, 5)
            x = conv2d(x, 16, 5, 3)
            x = conv2d(x, 8, 3, 1)

            x = tf.reshape(x, [self._batch_size * self._seq_length, -1])
            x = layers.fully_connected(x, 128)
            x = layers.fully_connected(x, 64)
            x = layers.fully_connected(x, self._target_size_cnn)

            self._initial_output = x

        # Visualize cnn outputs distribution in Tensorboard
        # with tf.name_scope('initial_output'):
        #     mean = tf.reduce_mean(self._initial_output)
        #     tf.summary.scalar('mean', mean)
        #
        #     stddev = tf.sqrt(tf.reduce_mean(tf.square(self._initial_output - mean)))
        #     tf.summary.scalar('stddev', stddev)
        #     tf.summary.histogram('histogram', self._initial_output)

        # ---- RNN ----
        # build inputs for RNN
        with tf.variable_scope("build_complete_input"):

            # tansform input_data to delta
            delta_x = VRU2Model.input_data_delta_tf(self._input_data)
            self._tf_input_data = delta_x

            # reshape cnn outputs back to (N, seq_length, target_size_cnn)
            x = tf.reshape(x, shape=[self._batch_size, self._seq_length, self._target_size_cnn])

            # concat cnn outputs with delta input, shape (batch_size, seq_length, feature_size + target_size_cnn)
            complete_input = VRU2Model.concat_tensor(delta_x, x)

            self._complete_input = complete_input

        def get_a_cell():
            # initialize a recurrent unit
            single_cell = rnn.GRUCell(num_units=self._rnn_size)

            # wrap a dropout layer if applicable
            if self._mode == 'train' and self._keep_prob < 1.0:
                single_cell = rnn.DropoutWrapper(cell=single_cell, output_keep_prob=self._keep_prob)

            return single_cell

        with tf.variable_scope("rnn"):

            cell = rnn.MultiRNNCell([get_a_cell() for _ in range(self._n_layers)])

            # initial cell state
            _initial_state = cell.zero_state(batch_size=self._batch_size, dtype=tf.float32)

            # dynamic rnn, output shape (batch_size, seq_length, rnn_size)
            rnn_output, self._final_state = tf.nn.dynamic_rnn(
                cell=cell,
                inputs=complete_input,
                sequence_length=self._sequence_length,
                initial_state=_initial_state,
                dtype=None,
                parallel_iterations=None,
                swap_memory=False,
                time_major=False,
                scope=None)

        with tf.variable_scope('fc_'):

            # reshape to = (batch_size * seq_length, rnn_size)
            rnn_output = tf.reshape(rnn_output, shape=[-1, self._rnn_size])

            # shape = (batch_size * seq_length, target_size)
            outputs = layers.fully_connected(rnn_output, self._target_size)

            # reshape to = (batch_size, seq_length, target_size)
            outputs = tf.reshape(outputs, shape=[self._batch_size, self._seq_length, self._target_size])
            self._outputs = outputs

        with tf.variable_scope('loss'):

            # transform target data to delta, shape=(batch_size, seq_length, discrete_steps, feature_size)
            self._target_data_delta = VRU2Model.target_data_delta_tf(self._input_data, self._target_data)

            # shape = (batch_size, seq_length, discrete_steps, feature_size)
            outputs = tf.reshape(outputs, shape=[-1, self._seq_length, self._discrete_steps, self._feature_size])
            self._predicted_outputs = outputs

            # calculate (x-x')^2, (y-y')^2
            sqaure_loss = tf.square(tf.subtract(outputs, self._target_data_delta))
            # calculate sqrt ((x-x')^2 + (y-y')^2)
            loss = tf.sqrt(tf.reduce_sum(sqaure_loss, 2))  # shape = (batch_size, discrete_steps)
            loss = tf.reduce_mean(loss)

            self._loss = loss

            # add summary operations
            tf.summary.scalar('loss', self._loss)
            self._summary_op = tf.summary.merge_all()

        # transform predicted delta to absolute
        with tf.variable_scope('predicted_final'):
            self._tf_predicted_outputs = VRU2Model.restore_delta_to_absolute(self._input_data, outputs)

        if self._mode == 'infer':
            return

        # used in training phase to update parameters
        with tf.variable_scope('optimizer'):
            self._train_op = tf.train.AdamOptimizer(learning_rate=self._lr).minimize(loss, global_step=self._global_step)

    @ staticmethod
    def target_data_delta_tf(inputs_tensor, target_tensor):
        """
        transform the target trajectory to its relative coordinates, relative to the input data
        return the 4D tensor with the same shape as target_tensor

        :param inputs_tensor: 3D tensor, shape (N, seq_length, feature_size)
        :param target_tensor: 4D tensor, shape (N, seq_length, discrete_steps, feature_size)
        :return: 4D tensor of transformed relative target coordinates,  (N, seq_length, discrete_steps, feature_size)
        """
        inputs_tensor_ = tf.expand_dims(inputs_tensor, dim=2)  # expand 3d to 4d
        # target_tensor= (N, seq_length, discrete_steps, feature_size)
        # inputs_tensor_= (N, seq_length, 1, feature_size)
        result_tensor = tf.subtract(target_tensor, inputs_tensor_)

        return result_tensor

    @ staticmethod
    def restore_delta_to_absolute(inputs_tensor, predicted_tensor):
        """
        restore the tensor(usually predicted trajectory), given the input_tensor, from
        its delta coordinates back to the absolute coordinates on context

        :param inputs_tensor: 4D tensor, shape (N, seq_length, discrete_steps, feature_size)
        :param predicted_tensor: 4D tensor, shape (N, seq_length, discrete_steps, feature_size)
        :return: restored predicted outputs 4D tensor, shape (N, seq_length , discrete_steps, feature_size)
        """

        # do the first batch sample
        batch_z = inputs_tensor[0]
        offset = tf.expand_dims(batch_z, axis=1)
        restored_tensor = tf.expand_dims(tf.add(predicted_tensor[0], offset), axis=0)

        # do the rest of the batch
        for i in range(batch_z - 1):
            i += 1

            # get the last point in sequence from the inputs_tensor
            offset = tf.expand_dims(inputs_tensor[i], axis=1)

            # target add offset to get absolute coordinates, shape (1, seq_length, discrete_steps, feature_size)
            target_tf = tf.expand_dims(tf.add(predicted_tensor[i], offset), axis=0)

            # concatenate each batch
            restored_tensor = tf.concat([restored_tensor, target_tf], axis=0)

        return restored_tensor

    @ staticmethod
    def input_data_delta_tf(inputs_tensor):
        """
        a function that transforms a given list of coordinates to its list of relative positions, relative to its
        previous point. the first coordinate is seen as the offset (0,0)

        :param inputs_tensor: 3D tensor, shape (N, seq_length, feature_size)
        :return: tensor, has the same shape as the inputs tensor
        """
        __, s_length, feature_z = inputs_tensor.shape

        # construct a tensor of ones
        zero_tensor = tf.convert_to_tensor(np.zeros(shape=(1, 1, feature_z)), dtype=tf.float32)

        # copy values starting from the second one from inputs_tensor
        # do the first batch
        target_tensor = tf.concat([zero_tensor,
                                   tf.expand_dims(inputs_tensor[0][0:s_length - 1], 0)], axis=1)
        # for the rest of the batch
        for i in range(inputs_tensor.shape[0]-1):
            i += 1

            y = tf.expand_dims(inputs_tensor[i][0:s_length - 1], axis=0)
            # copy second to last value point
            y = tf.concat([zero_tensor, y], axis=1)
            # add to batch
            target_tensor = tf.concat([target_tensor, y], axis=0)

        # subtract to get delta
        target_tensor = tf.subtract(inputs_tensor, target_tensor)

        return target_tensor

    @ staticmethod
    def concat_tensor(input_tensor, concat_tensor):
        """
        Concatenate concat_tensor to input_tensor along its 3rd dimension batch-wisely. The concat_tensor
        could be either 2D or 3D. if concat_tensor is 2D, an extra dimension will be added to its second dim.

        :param input_tensor: 3D tensor, shape (N, seq_length, any)
        :param concat_tensor:  2D tensor, shape (N, any)
                               or 3D tensor, shape (N, seq_length, any)
        :return: 3D tensor (N, seq_length, any)
        """

        # if dims don't match, add an extra to convert 2D to 3D
        if len(concat_tensor.shape) != len(input_tensor.shape):
            concat_tensor = tf.expand_dims(concat_tensor, dim=1)  # shape (batch_size, 1, ?)

        # build a tensor of ones to copy values from concat_tensor
        temp = tf.convert_to_tensor(np.ones(
            shape=(concat_tensor.shape[0], input_tensor.shape[1], concat_tensor.shape[-1])), dtype=tf.float32)

        # copy context value by multiplying with ones tensor with broadcasting
        context_reshaped = tf.multiply(concat_tensor, temp)  # shape (batch_size, seq_length, ?)

        # concat with input data
        target_concat = tf.concat([input_tensor, context_reshaped], axis=2)

        return target_concat

    @property
    def input_data(self):
        return self._input_data

    @property
    def tf_input_data(self):
        return self._tf_input_data

    @property
    def target_data(self):
        return self._target_data

    @property
    def target_data_delta(self):
        return self._target_data_delta

    @property
    def context_input(self):
        return self._context_input

    @property
    def outputs(self):
        return self._outputs

    @property
    def loss(self):
        return self._loss

    @property
    def train_op(self):
        return self._train_op

    @property
    def summary_op(self):
        return self._summary_op

    @property
    def initial_output(self):
        return self._initial_output

    @property
    def complete_input(self):
        return self._complete_input

    @property
    def predicted_outputs(self):
        return self._predicted_outputs

    @property
    def tf_predicted_outputs(self):
        return self._tf_predicted_outputs




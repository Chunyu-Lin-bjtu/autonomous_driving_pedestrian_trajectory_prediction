import tensorflow as tf
from tensorflow.contrib import rnn, layers
import numpy as np

import cv2
import math


class VMModel(object):
    """
    VM model:
    input: past trajectory points concatenated with z vector values (encoded for the corresponding context patch)
    output: predicted future trajectory of the next few timestamps
    """

    def __init__(self, args):
        """
        configure model parameters
        :param args: model configuration parameters
        """
        self.beta1 = 0.9
        self.beta2 = 0.999

        self._batch_size = args.batch_size
        self._feature_size = args.feature_size  # x, y
        self._mode = args.mode  # train or val

        # RNN parameters
        self._discrete_steps = len(args.discrete_timestamps)
        self._output_size = self._discrete_steps * args.feature_size

        self._rnn_size = args.rnn_size
        self._seq_length = args.seq_length
        self._n_layers = args.n_layers
        self._keep_prob = args.keep_prob
        self.learning_rate = args.learning_rate

        # past trajectory data, default shape (32, 64, 2)
        self._input_data = tf.placeholder(
            tf.float32, shape=[
                self._batch_size, self._seq_length, self._feature_size], name='input_data')

        # ground true future trajectory
        self._target_data = tf.placeholder(
            tf.float32, shape=[
                self._batch_size, self._seq_length, self._discrete_steps, self._feature_size], name='target_data')

        # target delta transformed from target data
        self._target_data_delta = tf.placeholder(
            tf.float32, shape=[
                self._batch_size, self._seq_length, self._discrete_steps, self._feature_size], name='target_data_delta')

        # z code for context image/patch
        self._z = tf.placeholder(
            tf.float32, shape=[self._batch_size, self._seq_length, args.n_z], name='neighbor_context_resize')

        # optimizer global step
        self._global_step = tf.Variable(0, trainable=False)
        # sequence length argument
        self._sequence_length = tf.convert_to_tensor([self._seq_length for _ in range(self._batch_size)])

        # ---- RNN ----

        # build inputs for RNN
        with tf.variable_scope("build_complete_input"):

            # transform input_data to delta position
            delta_x = self.tf_delta_between(self._input_data)
            self._tf_input_data = delta_x

            # concat delta input with z code, shape (batch_size, seq_length, feature_size + n_z)
            complete_input = self.concat_tensor(delta_x, self._z)

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

        # fc after rnn
        with tf.variable_scope('fc_'):
            # reshape to (batch_size * seq_length, rnn_size)
            rnn_output = tf.reshape(rnn_output, shape=[-1, self._rnn_size])
            # fc, shape (batch_size * seq_length, target_size)
            outputs = layers.fully_connected(rnn_output, self._output_size)

            # reshape to (batch_size, seq_length, target_size)
            outputs = tf.reshape(outputs,
                                 shape=[self._batch_size, self._seq_length, self._output_size])
            self._outputs = outputs

        with tf.variable_scope('loss'):
            # transform target placeholder data to relative
            # 4D (N, seq_length, discrete_step, feature_size)
            target_data_delta = self.tf_delta(self._input_data, self._target_data)

            # rescale target according to patch resize power
            self._target_data_delta = target_data_delta

            # 4D (N, seq_length, discrete_step, feature_size)
            outputs = tf.reshape(outputs, shape=[-1, self._seq_length, self._discrete_steps,
                                                 self._feature_size])
            self._predicted_outputs = outputs

            # compute loss
            # sqrt((x-x')^2 + (y-y')^2)
            sqaure_loss = tf.square(tf.subtract(outputs, self._target_data_delta))
            loss = tf.sqrt(tf.reduce_sum(sqaure_loss, -1))   # shape (batch_size, seq_length, discrete_steps)
            loss = tf.reduce_mean(loss)

            self._loss = loss

            # add summary operations
            tf.summary.scalar('loss', self._loss)
            self._summary_op = tf.summary.merge_all()

        with tf.variable_scope('predicted_final'):
            # shape = (batch_size, discrete_steps, feature_size)
            self._tf_predicted_outputs = self.restore_outputs_to_absolute(self._input_data, outputs)
        if self._mode == 'infer':
            return

        with tf.variable_scope('optimizer'):
            adam_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1, beta2=self.beta2)
            self._train_op = adam_op.minimize(loss, global_step=self._global_step)

    def tf_delta(self, inputs_tensor, target_tensor):
        """
        transform the target trajectory to its relative coordinates, relative to the input data
        return the 4D tensor with the same shape as target_tensor

        :param inputs_tensor: 3D tensor, shape (N, seq_length, feature_size)
        :param target_tensor: 4D tensor, shape (N, seq_length, discrete_steps, feature_size)
        :return: 4D tensor of transformed relative target coordinates,  (N, seq_length, discrete_steps, feature_size)
        """
        inputs_tensor_ = tf.expand_dims(inputs_tensor, dim=2)  # expand 3d to 4d

        # target_tensor: (N, seq_length, discrete_steps, feature_size)
        # inputs_tensor_: (N, seq_length, 1, feature_size)
        result_tensor = tf.subtract(target_tensor, inputs_tensor_)

        return result_tensor

    def restore_outputs_to_absolute(self, inputs_tensor, predicted_tensor):
        """
        restore the given tensor of relative coordinates back to coordinates on context

        :param inputs_tensor: 4D tensor, shape (N, seq_length, discrete_steps, feature_size)
        :param predicted_tensor: 4D tensor, shape (N, seq_length, discrete_steps, feature_size)
        :return: restored predicted outputs 4D tensor, shape (N, seq_length , discrete_steps, feature_size)
        """
        # do the first batch sample
        set_offset = tf.expand_dims(inputs_tensor[0], axis=1) # 64, 1, 2
        restored_tensor = tf.expand_dims(tf.add(predicted_tensor[0], set_offset), axis=0)

        # do the rest of the batch
        for i in range(self._batch_size - 1):
            i += 1

            # get the last point in sequence from the inputs_tensor
            offset = tf.expand_dims(inputs_tensor[i], axis=1)

            # target add offset to get absolute coordinates, shape (1, seq_length, discrete_steps, feature_size)
            target_tf = tf.expand_dims(tf.add(predicted_tensor[i], offset), axis=0)

            # concatenate each batch
            restored_tensor = tf.concat([restored_tensor, target_tf], axis=0)

        return restored_tensor

    def tf_delta_between(self, inputs_tensor):
        """
        a function that transforms a given list of coordinates to its list of relative positions, relative to its
        previous point. the first coordinate is seen as the offset (0,0)

        :param inputs_tensor: 3D tensor, shape (N, seq_length, feature_size)
        :return: tensor, has the same shape as the inputs tensor
        """
        # construct a tensor of ones
        zero_tensor = tf.convert_to_tensor(np.zeros(shape=(1, 1, self._feature_size)), dtype=tf.float32)

        # copy values start from the second one from inputs_tensor
        # do the  first batch
        target_tensor = tf.concat([zero_tensor,
                                   tf.expand_dims(inputs_tensor[0][0:self._seq_length - 1], 0)], axis=1)
        # do the rest of the batch
        for i in range(inputs_tensor.shape[0] - 1):
            i += 1

            y = tf.expand_dims(inputs_tensor[i][0:self._seq_length - 1], axis=0)
            # copy second to last value point
            y = tf.concat([zero_tensor, y], axis=1)
            # add to batch
            target_tensor = tf.concat([target_tensor, y], axis=0)

        # subtract to get delta x, y coordinate
        target_tensor = tf.subtract(inputs_tensor, target_tensor)

        return target_tensor

    def concat_tensor(self, input_tensor, concat_tensor):
        """
        Concatenate concat_tensor to input_tensor along the 3rd dimension, batch-wisely. The concat_tensor
        could be either 2D or 3D. if 2D, an extra dimension will be added to its second dim to match with
        the input_tensor and concatenate along with the third dimension.

        :param input_tensor: 3D tensor, shape (N, seq_length, any)
        :param concat_tensor:  2D tensor, shape (N, any)
                            or 3D tensor, shape (N, seq_length, any)
        :return: Tensor shape (N, seq_length, any)
        """

        # if dims don't match, add an extra to convert 2D to 3D
        if len(concat_tensor.shape) != len(input_tensor.shape):
            concat_tensor = tf.expand_dims(concat_tensor, dim=1)  # shape (batch_size, 1, ?)

        # a tensor of ones
        temp = tf.convert_to_tensor(np.ones(
            shape=(concat_tensor.shape[0], input_tensor.shape[1], concat_tensor.shape[-1])), dtype=tf.float32)

        # copy context value by multiplying with ones tensor with broadcasting
        context_reshaped = tf.multiply(concat_tensor, temp)     # shape (batch_size, seq_length, ?)

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
    def z(self):
        return self._z

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
    def predicted_outputs(self):
        return self._predicted_outputs

    @property
    def tf_predicted_outputs(self):
        return self._tf_predicted_outputs

    @property
    def complete_input(self):
        return self._complete_input

import matplotlib
matplotlib.use('Agg')

from data_loader import *
from vru_model import *
from vru_config import *

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()


def train(data_root_dir, context_files_root_dir, csv_files_root_dir, dump_model_para_root_dir):
    """
    Train phase main process

    :return:
    """
    args = set_train_args()

    dataloader = DataLoader(
        data_root_dir=data_root_dir,  #'../../dataset_data/'
        train_batch_size=args.batch_size,
        val_batch_size=args.batch_size,
        seq_length=args.seq_length,
        frequency=args.frequency,
        discrete_timestamps=args.discrete_timestamps,
        context_files_root_dir=context_files_root_dir,
        csv_files_root_dir=csv_files_root_dir
    )

    print '\nDone loading datasets! \n\nReady to train.'

    # create model
    vrumodel = VRUModel(args)

    print 'Model initialized successfully.'
    # to save log
    train_error = np.zeros(args.num_epochs)
    valid_error = np.zeros(args.num_epochs)

    step = 0

    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(args.train_summary)
        val_writer = tf.summary.FileWriter(args.val_summary)

        # initialize all variables in the graph
        tf.global_variables_initializer().run()

        # initialize a saver that saves all the variables in the graph
        saver = tf.train.Saver(max_to_keep=None)
 
        print 'Start Training... \n'
        for e in range(args.num_epochs):

            dataloader.reset_train_pointer()
            dataloader.reset_val_pointer()

            # --- TRAIN ---
            for batch in range(dataloader.train_batch_num):

                # context data shape: (batch_size, width*height, )
                # input batch shape: (batch_size, seq_length, feature_size)
                # target batch shape: (batch_size, discrete_steps, feature_size)
                (context_data, input_batch), target_batch = dataloader.next_batch(mode='train')

                # reshape to feed in network [batch_size, in_width, in_height, in_channel], add color channel
                context_data_feed = context_data.reshape((-1, args.width, args.height, 1))

                batch_error_train, summary, __ = sess.run(
                    fetches=[
                        #vrumodel.initial_output
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
                (context_data, input_batch), target_batch = dataloader.next_batch(mode='val')

                context_data = context_data.reshape((-1, args.width, args.height, 1))

                summary, predicted_outputs, batch_error_val = sess.run(
                    fetches=[vrumodel.summary_op, vrumodel.predicted_outputs, vrumodel.loss],
                    feed_dict={
                        vrumodel.input_data: input_batch,
                        vrumodel.target_data: target_batch,
                        vrumodel.context_input: context_data
                    })
                #print batch_error_val
                val_writer.add_summary(summary, step)
                valid_error[e] += batch_error_val
                step += 1
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

            # # plot loss
            # fig = plt.figure()
            # ax = fig.add_subplot()
            # plt.plot(train_error)
            # plt.plot(valid_error)
            # plt.show()
            #
            # plt.pause(0.001)  # pause a bit so that plots are updated
            # if is_ipython:
            #     display.clear_output(wait=True)
            #     display.display(plt.gcf())

    # close writer and session objects
    train_writer.close()
    val_writer.close()
    sess.close()

    return train_error, valid_error


if __name__ == '__main__':

    train_error, valid_error = train(data_root_dir='../../dataset_data/',
          context_files_root_dir='../../dataset/context/',
          csv_files_root_dir='../../dataset/trajectory/',
          dump_model_para_root_dir='../vru_model/')
    plt.plot(train_error)
    plt.plot(valid_error)
    plt.show()
    plt.savefig('vru_loss.png')

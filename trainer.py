import time
import models
# from models import *
from running_logger.create_logger import *
from datasets.data_preprocess import *
# from datasets.record_dataset import *
from tensorflow.python.framework import graph_util
from tensorboardX import SummaryWriter

slim = tf.contrib.slim


class Trainer():
    def __init__(self, output_path=None, config=None):
        self.output_path = output_path
        self.config = config
        self.logger = Logger(log_file_name=output_path + '/log.txt',
                             log_level=logging.DEBUG, logger_name="").get_log()
        # start a SummaryWriter
        self.writer = SummaryWriter(log_dir=self.output_path + '/event')
        # self.writer = tf.summary.FileWriter(self.output_path + '/event', flush_secs=60)
        self.image_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, config.input_resize_h,
                                                                         config.input_resize_w,
                                                                         config.input_size_d], name='inputs')
        self.label_placeholder = tf.placeholder(dtype=tf.int32, shape=[None])
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.best_test_acc = 0
        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_device
        if config.architecture == 'simple_model':
            self.cls_model = models.simple_model.Model(is_training=self.is_training, num_classes=config.num_classes)
        elif config.architecture == 'resnet_v1_50':
            if config.dataset == 'cifar10' or 'captcha':
                self.cls_model = models.resnet_v1_50.Model(is_training=self.is_training, num_classes=config.num_classes,
                                                           fixed_resize_side=config.input_resize_h,
                                                           default_image_size=config.input_resize_h,
                                                           dataset_config=self.config)
            else:
                self.cls_model = models.resnet_v1_50.Model(is_training=True, num_classes=self.config.num_classes,
                                                           dataset_config=self.config)
        elif config.architecture == 'resnet_v1_101':
            if config.dataset == 'cifar10' or 'captcha':
                self.cls_model = models.resnet_v1_101.Model(is_training=self.is_training,
                                                            num_classes=config.num_classes,
                                                            fixed_resize_side=config.input_resize_h,
                                                            default_image_size=config.input_resize_h,
                                                            dataset_config=self.config)
            else:
                self.cls_model = models.resnet_v1_101.Model(is_training=True, num_classes=self.config.num_classes,
                                                            dataset_config=self.config)

    def start_train_and_val(self):
        # global acc_ave

        if self.config.architecture == 'resnet_v1_50' or 'resnet_v1_101':
            preprocessed_inputs = self.image_placeholder
        else:
            preprocessed_inputs = self.cls_model.preprocess(self.image_placeholder)
        prediction_dict = self.cls_model.predict(preprocessed_inputs)
        regu_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss_dict = self.cls_model.loss(prediction_dict, self.label_placeholder)
        loss_ = loss_dict['loss']
        loss = tf.add_n([loss_] + regu_losses)
        postprocessed_dict = self.cls_model.postprocess(prediction_dict)
        classes = postprocessed_dict['classes']
        classes = tf.cast(classes, tf.int32)
        tf.identity(classes, name='classes')
        train_acc = []
        test_acc = []
        train_acc_ = np.mean(train_acc)
        test_acc_ = np.mean(test_acc)
        acc = tf.reduce_mean(tf.cast(tf.equal(classes, self.label_placeholder), 'float'))
        global_step = tf.Variable(0, trainable=False)
        # learning_rate
        learning_rate = tf.train.exponential_decay(self.config.lr_scheduler.init_lr, global_step, 2500, 0.9)
        # optimizer
        optimizer = tf.train.MomentumOptimizer(learning_rate, self.config.optimize.momentum)
        # optimizer = tf.train.AdamOptimizer(learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss)
        # 如有BN，算子切勿使用次方式更新
        # train_op = optimizer.minimize(loss, global_step=global_step)

        # summary
        # train_loss_scalar = tf.summary.scalar('train_loss', loss)
        # acc_ave = tf.Variable(10, trainable=False, dtype="float")
        # trian_accuracy_scalar = tf.summary.scalar('trian_accuracy', acc_ave)
        # test_accuracy_scalar = tf.summary.scalar('test_accuracy', test_acc_)
        # lr_scalar = tf.summary.scalar('lr', learning_rate)

        saver = tf.train.Saver(max_to_keep=3)

        # 读取预训练模型用
        checkpoint_exclude_scopes = 'Logits'
        exclusions = None
        if checkpoint_exclude_scopes:
            exclusions = [
                scope.strip() for scope in checkpoint_exclude_scopes.split(',')]
        variables_to_restore = []
        for var in slim.get_model_variables():
            excluded = False
            for exclusion in exclusions:
                if var.op.name.startswith(exclusion):
                    excluded = True
            if not excluded:
                variables_to_restore.append(var)
        saver_restore = tf.train.Saver(var_list=variables_to_restore)

        # train_data, train_labels = read_train_data(self.config)
        # val_data, val_labels = read_validation_data(self.config)
        #
        # np.save("train_data.npy", train_data)
        # np.save("train_labels.npy", train_labels)
        # np.save("val_data.npy", val_data)
        # np.save("val_labels.npy", val_labels)

        train_data = np.load("train_data.npy")
        train_labels = np.load("train_labels.npy")
        val_data = np.load("val_data.npy")
        val_labels = np.load("val_labels.npy")

        # init = tf.global_variables_initializer()
        # init = tf.initialize_all_vasriables()
        with tf.Session() as sess:

            # 读预训练模型用
            saver_restore.restore(sess, self.config.ckpt_pretrain_path)

            # sess.run(init)
            if self.config.if_resume:
                saver.restore(sess, self.config.ckpt_resume_path)
                print('Restored from checkpoint...')
            else:
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())
            # coord = tf.train.Coordinator()
            # threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            # self.writer.add_graph(sess.graph)
            # train_acc_list = []
            for i in range(self.config.train_steps):
                # train
                batch_images, batch_labels = generate_augment_train_batch(train_data, train_labels, self.config)

                train_dict = {self.image_placeholder: batch_images, self.label_placeholder: batch_labels,
                              self.is_training: True}
                _, loss_, acc_, lr_ = sess.run(
                    [train_op, loss, acc, learning_rate],
                    feed_dict=train_dict)

                train_acc.append(acc_)

                # self.writer.add_summary(train_loss_scalar_, global_step=i)
                # if i > 100 and i % 1000 == 0:

                if i >= 1000 and i % 1000 == 0:
                    acc_ave = np.mean(train_acc)
                    self.writer.add_scalar('acc/train_acc', acc_ave, i)
                    # train_acc_scalar_ = sess.run(trian_accuracy_scalar)
                    # self.writer.add_summary(train_acc_scalar_, global_step=i)
                    train_acc = []

                # self.writer.add_summary(lr_scalar_, global_step=i)
                if i % 100 == 0:
                    train_text = 'step: {}, lr: {:.5f}, loss: {:.5f}, acc: {:.3f}'.format(
                        i + 1, lr_, loss_, acc_)
                    # print(train_text)
                    self.logger.info(train_text)
                # val
                if i > 100 and i % 1000 == 0:
                    if not os.path.exists(self.config.ckpt_path):
                        os.mkdir(self.config.ckpt_path)
                    # saver.save(sess, self.config.ckpt_path + str(i) + 'model.ckpt')

                    # for tb show | pic is tensor |img in train
                    image_shaped_input = tf.reshape(batch_images,
                                                    [-1, self.config.input_resize_w, self.config.input_resize_h,
                                                     self.config.input_size_d])
                    # 3 pics
                    # image_summary = tf.summary.image('image', image_shaped_input, 3)
                    # image_summary_ = sess.run(image_summary)
                    # self.writer.add_summary(image_summary_)

                    # image = tf.expand_dims(image_shaped_input, 0)
                    num_batches = self.config.val_num // self.config.val_batch
                    order = np.random.choice(self.config.val_num, num_batches * self.config.val_batch)
                    vali_data_subset = val_data[order, ...]
                    vali_labels_subset = val_labels[order]
                    loss_list = []
                    acc_list = []
                    self.logger.info('×*×*×*×*×*×*×*×*×*×*×*Start test×*×*×*×*×*×*×*×*×*×*×*')
                    start_time = time.time()
                    for step in range(num_batches):
                        offset = step * self.config.val_batch
                        val_feed_dict = {self.image_placeholder: vali_data_subset[
                                                                 offset:offset + self.config.val_batch, ...],
                                         self.label_placeholder: vali_labels_subset[
                                                                 offset:offset + self.config.val_batch],
                                         self.is_training: False
                                         }
                        # test_summary: test_loss_scalar, test_accuracy_scalar
                        val_loss, val_acc = sess.run(
                            [loss, acc], feed_dict=val_feed_dict)
                        loss_list.append(val_loss)
                        acc_list.append(val_acc)
                    time_count = time.time() - start_time
                    examples_per_sec = self.config.val_num / time_count
                    val_text = 'val_loss: {:.5f}, speed:{:.2f}iters/s, val_acc: {:.3}'.format(np.mean(loss_list),
                                                                                              examples_per_sec,
                                                                                              np.mean(acc_list))
                    if np.mean(acc_list) >= self.best_test_acc:
                        self.best_test_acc = np.mean(acc_list)
                        saver.save(sess, self.config.ckpt_path + 'best_model.ckpt')
                    # self.writer.add_summary(np.mean(acc_list), global_step=i)
                    self.writer.add_scalar('acc/test_acc', np.mean(acc_list), i)
                    self.logger.info(val_text)
            # if not os.path.exists(self.config.ckpt_path):
            #     os.mkdir(self.config.ckpt_path)
            # saver.save(sess, self.config.ckpt_path + str(i) + 'model.ckpt')
            if self.config.save_pb_direct:
                graph_def = tf.get_default_graph().as_graph_def()
                output_graph_def = graph_util.convert_variables_to_constants(
                    sess,
                    graph_def,
                    ['classes']
                )
                with tf.gfile.GFile(self.config.pb_direct_path + 'model.pb', 'wb') as fid:
                    serialized_graph = output_graph_def.SerializeToString()
                    fid.write(serialized_graph)
            print("Trianing Finished!")
            # coord.request_stop()
            # coord.join(threads)

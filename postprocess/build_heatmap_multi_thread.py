from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import threading
import os.path
import time
from datetime import datetime
import math
from PIL import Image
from matplotlib import pyplot as plt
import cv2
import numpy as np
import tensorflow as tf

sys.path.append('../')
import utils as utils
from inception_v3 import image_processing
from inception_v3 import inception_model as inception
from inception_v3.dataset import Dataset
from inception_v3.slim import slim

#********************************************************************************
# CKPT_PATH = utils.EVAL_MODEL_CKPT_PATH
DATA_SET_NAME = 'TF-Records'

#Flags govering the training and evaluation logs
tf.app.flags.DEFINE_string('eval_dir', utils.EVAL_LOGS,
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', utils.TRAIN_MODELS,
                           """Directory where to read model checkpoints.""")

# Flags governing the frequency of the eval.
tf.app.flags.DEFINE_integer('num_threads', 2,
                            """Number of threads.""")
tf.app.flags.DEFINE_boolean('run_once', True,
                            """Whether to run eval only once.""")

# Flags governing the data used for the eval.
tf.app.flags.DEFINE_integer('num_examples', 1,
                            """Number of examples to run.
                            We have 1 examples.""")
tf.app.flags.DEFINE_string('subset', 'heatmap',
                           """Either 'validation' or 'train'.""")

# tf.app.flags.DEFINE_integer('batch_size', 40,
                            # """Number of images to process in a batch.""")

FLAGS = tf.app.flags.FLAGS
BATCH_SIZE = 50

#**************************************************************************
def evaluate_split(thread_index, sess, prob_ops, cords):
    global heat_map
    print('evaluate_split(): thread-%d' % thread_index)
    probabilities, coordinates = sess.run([prob_ops, cords])

    for prob, cord in zip(probabilities, coordinates):
        cord = cord[0].decode('UTF-8')
        pixel_pos = cord.split('_')
        heat_map[int(pixel_pos[0]), int(pixel_pos[1])] = prob[0]

#***************************************************************************
def generate_heatmap(saver, dataset, summary_writer, prob_ops, cords_ops, summary_op):
    # def _eval_once(saver, summary_writer, accuracy, summary_op, confusion_matrix_op, logits, labels, dense_labels):
    # with tf.Session() as sess:
    #     ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    #     if CKPT_PATH is not None:
    #         saver.restore(sess, CKPT_PATH)
    #         global_step = CKPT_PATH.split('/')[-1].split('-')[-1]
    #         print('Successfully loaded model from %s at step=%s.' %
    #               (CKPT_PATH, global_step))
    #     elif ckpt and ckpt.model_checkpoint_path:
    #         print(ckpt.model_checkpoint_path)
    #         if os.path.isabs(ckpt.model_checkpoint_path):
    #             # Restores from checkpoint with absolute path.
    #             saver.restore(sess, ckpt.model_checkpoint_path)
    #         else:
    #             # Restores from checkpoint with relative path.
    #             saver.restore(sess, os.path.join(FLAGS.checkpoint_dir,
    #                                              ckpt.model_checkpoint_path))
    #
    #         # Assuming model_checkpoint_path looks something like:
    #         #   /my-favorite-path/imagenet_train/model.ckpt-0,
    #         # extract global_step from it.
    #         global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    #         print('Successfully loaded model from %s at step=%s.' %
    #               (ckpt.model_checkpoint_path, global_step))
    #     else:
    #         print('No checkpoint file found')
    #         return
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        # global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        print("the model path is :%s" % ckpt.model_checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("successfully load model from path")
        else:
            print('No checkpoint file found')
            return
        # Start the queue runners.
        coord1 = tf.train.Coordinator()
        coord2 = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord1, daemon=True,
                                                 start=True))

            num_iter = int(math.ceil(dataset.num_examples_per_epoch() / BATCH_SIZE))
            step = 0
            print('%s: starting evaluation on (%s).' % (datetime.now(), FLAGS.subset))
            start_time = time.time()

            while step < num_iter and not coord1.should_stop():
                eval_threads = []
                for thread_index in range(FLAGS.num_threads):
                    args = (thread_index, sess, prob_ops[thread_index], cords_ops[thread_index])
                    t = threading.Thread(target=evaluate_split, args=args)
                    t.start()
                    eval_threads.append(t)
                    # 调试需要关闭多线程
                    # evaluate_split(thread_index,sess,prob_ops[thread_index],cords_ops[thread_index])


                coord2.join(eval_threads)
                step += 1
                print('%s: patch processed: %d / %d' % (datetime.now(), step * BATCH_SIZE,
                                                        dataset.num_examples_per_epoch()))
                if not ((step * BATCH_SIZE) % 1000):
                    duration = time.time() - start_time
                    print('1000 patch process time: %d secs' % math.ceil(duration))
                    start_time = time.time()

        except Exception as e:  # pylint: disable=broad-except
            coord1.request_stop(e)

        coord1.request_stop()
        coord1.join(threads, stop_grace_period_secs=10)

#**************************************************************************
def build_heatmap(dataset):
    """Evaluate model on Dataset for a number of steps."""
    with tf.Graph().as_default():
        # Get images and labels from the dataset.
        images, cords = image_processing.inputs(dataset, BATCH_SIZE) #cords is label?
        print('images process is done')
        # Number of classes in the Dataset label set plus 1.
        # Label 0 is reserved for an (unused) background class.
        num_classes = dataset.num_classes()
        assert BATCH_SIZE % FLAGS.num_threads == 0, 'BATCH_SIZE must be divisible by FLAGS.num_threads'
        # Build a Graph that computes the logits predictions from the
        # inference model.
        images_splits = tf.split(images, FLAGS.num_threads, axis=0)
        cords_splits = tf.split(cords, FLAGS.num_threads, axis=0)
        prob_ops = []
        cords_ops = []
        for i in range(FLAGS.num_threads):
            with tf.name_scope('%s_%d' % (inception.TOWER_NAME, i)) as scope:
                with slim.arg_scope([slim.variables.variable], device='/cpu:%d' % i):
                    print('i=%d' % i)

                    _, _, prob_op = inception.inference(images_splits[i], num_classes, scope=scope)
                    tf.get_variable_scope().reuse_variables()
                    cords_op = tf.reshape(cords_splits[i], (int(BATCH_SIZE/FLAGS.num_threads), 1))
                    prob_ops.append(prob_op)
                    cords_ops.append(cords_op)
        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            inception.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()
        graph_def = tf.get_default_graph().as_graph_def()
        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, graph_def=graph_def)
        generate_heatmap(saver, dataset, summary_writer, prob_ops, cords_ops, summary_op)

#***********************************************************************************
def main(unused_argv):
    global heat_map
    # global heat_map_prob

    tf_records_file_names = sorted(os.listdir(utils.HEAT_MAP_TF_RECORDS_DIR))
    print(tf_records_file_names)
    # tf_records_file_names = tf_records_file_names[2:3]
    for wsi_filename in tf_records_file_names:
        prefix = wsi_filename.split('_')[0]
        if not prefix ==  'Test':
            print('jump out of %s' %wsi_filename)
            continue
        print('Generating heatmap for: %s' % wsi_filename)
        tf_records_dir = os.path.join(utils.HEAT_MAP_TF_RECORDS_DIR, wsi_filename)

        #------这个地方可以做个优化，把patches的数量提前保存起来，就不用再计算了--------------
        raw_patches_dir = os.path.join(utils.HEAT_MAP_RAW_PATCHES_DIR, wsi_filename)
        #----------------------------------------------------------------------------
        #-----------提前把wsi image的尺寸保存下来---------------------------------------
        wsi_img_path = utils.HEAT_MAP_WSIs_PATH + '/' + wsi_filename + 'lowest_level.png' 
        assert os.path.exists(wsi_img_path), 'heatmap rgb image %s does not exist' % wsi_img_path
        wsi_img = Image.open(wsi_img_path)
        wsi_img = np.array(wsi_img)
        wsi_img = wsi_img[:, :, :1]
        wsi_img = np.reshape(wsi_img, (wsi_img.shape[0], wsi_img.shape[1]))
        heat_map = np.zeros((wsi_img.shape[0], wsi_img.shape[1]), dtype=np.float32)
        # heat_map_prob = np.zeros((wsi_img.shape[0], wsi_img.shape[1]), dtype=np.float32)
        #----------------------------------------------------------------------------------
        #------这个地方可以做个优化，把patches的数量提前保存起来，就不用再计算了--------------
        assert os.path.exists(raw_patches_dir), 'raw patches directory %s does not exist' % raw_patches_dir
        num_patches = len(os.listdir(raw_patches_dir))
        #----------------------------------------------------------------------------

        assert os.path.exists(tf_records_dir), 'tf-records directory %s does not exist' % tf_records_dir
        dataset = Dataset(DATA_SET_NAME, utils.data_subset[4], tf_records_dir=tf_records_dir, num_patches=num_patches)
        
        build_heatmap(dataset)
        np.save(utils.HEAT_MAP_DIR + '/' + wsi_filename, heat_map)
        print('Finished')
        # Image.fromarray(heat_map).save(os.path.join(utils.HEAT_MAP_DIR, wsi_filename), 'PNG')
        # plt.imshow(heat_map, cmap='jet', interpolation='nearest')
        # plt.colorbar()
        # plt.clim(0.00, 1.00)
        # plt.axis([0, wsi_img.shape[1], 0, wsi_img.shape[0]])
        # plt.savefig(str(os.path.join(utils.HEAT_MAP_DIR, wsi_filename))+'_heatmap.png')
        # heatmap_prob_name_postfix = '_prob.png'
        # cv2.imwrite(os.path.join(utils.HEAT_MAP_DIR, wsi_filename) + heatmap_prob_name_postfix, heat_map_prob * 255)
        # plt.show()

if __name__ == '__main__':
    heat_map = None
    # heat_map_prob = None
    tf.app.run()

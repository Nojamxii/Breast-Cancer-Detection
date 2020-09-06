# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A library to evaluate Inception on a single GPU.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import math
import os.path
import time
from datetime import datetime
import glob

from inception_v3 import image_processing
from inception_v3 import inception_model as inception
import utils as utils
import numpy as np
import sklearn as sk
import tensorflow as tf
from inception_v3.dataset import Dataset
from tensorflow.contrib import metrics

FLAGS = tf.app.flags.FLAGS

CKPT_PATH = utils.EVAL_MODEL_CKPT_PATH

DATA_SET_NAME = 'TF-Records'



tf.app.flags.DEFINE_string('eval_logs', utils.EVAL_LOGS,
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', utils.TRAIN_MODELS,
                           """Directory where to read model checkpoints.""")

# Flags governing the frequency of the eval.
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_boolean('run_once', True,
                            """Whether to run eval only once.""")

# Flags governing the data used for the eval.
tf.app.flags.DEFINE_integer('num_examples', 444,
                            """Number of examples to run.
                            We have 10000 examples.""")
tf.app.flags.DEFINE_string('subset', 'validation',
                           """Either 'validation' or 'train'.""")

# tf.app.flags.DEFINE_integer('batch_size', 40,
#                             """Number of images to process in a batch.""")

BATCH_SIZE = 50


def _eval_once(saver, summary_writer, accuracy, summary_op, confusion_matrix_op,labels,logits):
    # def _eval_once(saver, summary_writer, accuracy, summary_op, confusion_matrix_op, logits, labels, dense_labels):

    """Runs Eval once.

    Args:
      saver: Saver.Restore the moving average version of the learned variables for eval.
      summary_writer: Summary writer.
      top_1_op: Top 1 op.
      top_5_op: Top 5 op.
      summary_op: Summary op.
    """
    """
    # with tf.Session() as sess:
    #     print(FLAGS.checkpoint_dir)#在train里
    #     ckpt = None
    #     if CKPT_PATH is not None:
    #         saver.restore(sess, CKPT_PATH)
    #         global_step = CKPT_PATH.split('/')[-1].split('-')[-1]
    #         print('Succesfully loaded model from %s at step=%s.' %
    #               (CKPT_PATH, global_step))
    #     elif ckpt is None:
    #         ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    #         if ckpt and ckpt.model_checkpoint_path:
    #             print(ckpt.model_checkpoint_path)
    #             if os.path.isabs(ckpt.model_checkpoint_path):
    #                 # Restores from checkpoint with absolute path.
    #                 saver.restore(sess, ckpt.model_checkpoint_path)
    #             else:
    #                 # Restores from checkpoint with relative path.
    #                 saver.restore(sess, os.path.join(FLAGS.checkpoint_dir,
    #                                                  ckpt.model_checkpoint_path))
    # 
    #             # Assuming model_checkpoint_path looks something like:
    #             #   /my-favorite-path/imagenet_train/model.ckpt-0,
    #             # extract global_step from it.
    #             global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    #             print('Succesfully loaded model from %s at step=%s.' %
    #                   (ckpt.model_checkpoint_path, global_step))
    #     else:
    #         print('No checkpoint file found')
    #         return
    #     """

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        print("the model path is :%s" % ckpt.model_checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("successfully load model from path")
            print("global step:")
            print(global_step)
        else:
            print('No checkpoint file found')
            return

        # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                 start=True))

            num_iter = int(math.ceil(FLAGS.num_examples / BATCH_SIZE))
            # Counts the number of correct predictions.
            total_correct_count = 0
            total_false_positive_count = 0
            total_false_negative_count = 0
            total_true_positive_count = 0
            total_true_negative_count = 0
            total_sample_count = num_iter * BATCH_SIZE
            step = 0
            total_predict_label = []

            print('%s: starting evaluation on (%s).' % (datetime.now(), FLAGS.subset))
            start_time = time.time()
            while step < num_iter and not coord.should_stop():
                label,logit,correct_count, confusion_matrix = sess.run([labels,logits,accuracy, confusion_matrix_op])
                # print("step:%d" % step)
                # print("label:")
                # print(label)

                # print('logit:')
                # print(logit)
                # print('accuracy:')
                # print(correct_count)
                # correct_count, confusion_matrix, logits_v, labels_v, dense_labels_v = sess.run([accuracy, confusion_matrix_op, logits, labels, dense_labels])
                # 合并所有的prdict label
                predict_lebel = list(correct_count)
                total_predict_label += predict_lebel

                total_correct_count += np.sum(correct_count)
                # total_false_positive_count += confusion_matrix[1][0]
                # total_false_negative_count += confusion_matrix[0][1]
                # total_true_positive_count +=confusion_matrix[1][1]
                # total_true_negative_count +=confusion_matrix[0][0]
                print('confusion_matrix:')
                print(confusion_matrix)
                # print('total_false_positive_count: %d' % total_false_positive_count)
                # print('total_false_negative_count: %d' % total_false_negative_count)
                # print('total_true_positive_count: %d' % total_true_positive_count)
                # print('total_true_negative_count: %d' % total_true_negative_count)
                print('correct_count(step=%d): %d / %d' % (step, total_correct_count, BATCH_SIZE * (step + 1)))


                step += 1
                if step % 20 == 0:
                    duration = time.time() - start_time
                    sec_per_batch = duration / 20.0
                    examples_per_sec = BATCH_SIZE / sec_per_batch
                    print('%s: [%d batches out of %d] (%.1f examples/sec; %.3f'
                          'sec/batch)' % (datetime.now(), step, num_iter,
                                          examples_per_sec, sec_per_batch))
                    start_time = time.time()

            # print('total_false_positive_count: %d' % total_false_positive_count)
            # print('total_false_negative_count: %d' % total_false_negative_count)
            # Compute precision @ 1.


            # print('total pridict label:')
            # print(total_predict_label)

            wsi_path = glob.glob(os.path.join(utils.TRAIN_TUMOR_WSI_PATH, '*.tif'))
            wsi_path.sort()
            tumor_name = []
            for wsi_name in wsi_path:
                tumor_path = utils.get_filename_from_path(wsi_name).split('.')[0]
                tumor_name.append(tumor_path)
            # print(tumor_name)

            start_number = 0
            for tests in tumor_name:
                patch_name = '*' + tests + '.png'
                patches_paths = glob.glob(os.path.join('E:/2016/Validation-Set/Extracted_Positive_Patches', patch_name))
                patches_paths.sort()
                patches_number = len(patches_paths)
                print(patch_name)
                end_number = start_number+patches_number
                each_wsi_correct_number = np.sum(total_predict_label[start_number:end_number])
                print('%d/%d'%(each_wsi_correct_number,patches_number))
                print('the correct value is :%.4f' % (each_wsi_correct_number/patches_number))
                start_number+=patches_number

            precision = total_correct_count / total_sample_count
            print('%s: precision = %.4f [%d examples]' % (datetime.now(), precision, total_sample_count))

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Precision', simple_value=precision)
            summary_writer.add_summary(summary, global_step)

        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def calc_metrics(dense_labels, logits):
    print("Precision", sk.metrics.precision_score(dense_labels, logits))
    print("Recall", sk.metrics.recall_score(dense_labels, logits))
    print("f1_score", sk.metrics.f1_score(dense_labels, logits))
    print("confusion_matrix")
    print(sk.metrics.confusion_matrix(dense_labels, logits))


def evaluate(dataset):
    """Evaluate model on Dataset for a number of steps."""
    with tf.Graph().as_default():
        # Get images and labels from the dataset.
        images, labels = image_processing.inputs(dataset, BATCH_SIZE)
        print("image processing is done.")

        # Number of classes in the Dataset label set plus 1.
        # Label 0 is reserved tf.get_variable_scope()for an (unused) background class.
        num_classes = dataset.num_classes()

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits, _, _ = inception.inference(images, num_classes)

        # sparse_labels = tf.reshape(labels, [BATCH_SIZE, 1])
        # indices = tf.reshape(tf.range(BATCH_SIZE), [BATCH_SIZE, 1])
        # concated = tf.concat(1, [indices, sparse_labels])
        # num_classes = logits[0].get_shape()[-1].value
        # dense_labels = tf.sparse_to_dense(concated,[BATCH_SIZE, num_classes],1, 0)


        #argmax返回沿着某个维度最大值的位置#
        confusion_matrix_op = metrics.confusion_matrix(labels, tf.argmax(logits, axis=1))
        # false_positive_op = metrics.streaming_false_positives(logits, dense_labels)
        # false_negative_op = metrics.streaming_false_negatives(logits, dense_labels)

        # Calculate predictions.
        '''predictions：预测的结果，预测矩阵大小为样本数×标注的label类的个数的二维矩阵。targets：实际的标签，大小为样本数。k：每个样本的预测结果的前k个最大的数里面是否包含targets
        预测中的标签，一般都是取1，即取预测最大概率的索引与标签对比。name：名字。 '''
        accuracy = tf.nn.in_top_k(logits, labels, 1)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(inception.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        graph_def = tf.get_default_graph().as_graph_def()
        summary_writer = tf.summary.FileWriter(FLAGS.eval_logs, graph_def=graph_def)

        while True:
            # _eval_once(saver, summary_writer, accuracy, summary_op, confusion_matrix_op, logits, labels, dense_labels)

            _eval_once(saver, summary_writer, accuracy, summary_op, confusion_matrix_op,labels,logits)
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)



def main(unused_argv):
    dataset = Dataset(DATA_SET_NAME, utils.data_subset[2])
    evaluate(dataset)

if __name__ == '__main__':
    tf.app.run()
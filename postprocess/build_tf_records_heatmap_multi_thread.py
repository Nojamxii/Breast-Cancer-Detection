import sys
import os
import time
from datetime import datetime
import math
import tensorflow as tf
import threading
import glob
import utils as utils
import shutil

N_TRAIN_SAMPLES = 250000
N_VALIDATION_SAMPLES = 10000
N_SAMPLES_PER_TRAIN_SHARD = 1000
N_SAMPLES_PER_VALIDATION_SHARD = 250

tf.app.flags.DEFINE_string('output_directory', utils.HEAT_MAP_TF_RECORDS_DIR,
                           'Output data directory')

tf.app.flags.DEFINE_integer('num_shards', 2,
                            'Number of shards in training TFRecord files.')

tf.app.flags.DEFINE_integer('num_threads', 1,
                            'Number of threads to preprocess the images.')

FLAGS = tf.app.flags.FLAGS


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(image_buffer, patch_name):
    """
        Build an Example proto for an example.

    """

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/patch_name': _bytes_feature(tf.compat.as_bytes(patch_name)),
        'image/encoded': _bytes_feature(tf.compat.as_bytes(image_buffer))}))
    return example


class ImageCoder(object):
    """Helper class that provides tf image coding utilities."""

    def __init__(self):
        # Create a single Session to run all image coding calls.
        self._sess = tf.Session()

        # Initializes function that converts PNG to JPEG data.
        self._png_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

        self._decode_png_data = tf.placeholder(dtype=tf.string)
        self._decode_png = tf.image.decode_png(self._decode_png_data, channels=3)

    def png_to_jpeg(self, image_data):
        return self._sess.run(self._png_to_jpeg,
                              feed_dict={self._png_data: image_data})

    def decode_jpeg(self, image_data):
        image = self._sess.run(self._decode_jpeg,
                               feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image

    def decode_png(self, image_data):
        image = self._sess.run(self._decode_png,
                               feed_dict={self._decode_png_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def _is_png(filename):
    """Determine if a file contains a PNG format image.

    Args:
      filename: string, path of the image file.

    Returns:
      boolean indicating if the image is a PNG.
    """
    return '.png' in filename


def _process_image(patch_path, coder):
    """Process a single image file.

    Args:
      filename: string, path to an image file e.g., '/path/to/example.JPG'.
      coder: instance of ImageCoder to provide tf image coding utils.
    Returns:
      image_buffer: string, JPEG encoding of RGB image.
      height: integer, image height in pixels.
      width: integer, image width in pixels.
    """
    # Read the image file. pay attantion to the version of python 'r'->'rb'
    with tf.gfile.FastGFile(patch_path, 'rb') as f:
        image_data = f.read()

    # Decode the RGB PNG.
    image = coder.decode_png(image_data)

    # Check that image converted to RGB
    # assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    # assert image.shape[2] == 3

    return image_data, height, width


def _process_patches(thread_index, name, patch_paths, patch_names, wsi_filename):
    """Process and save list of images as TFRecord of Example protos.

    Args:
      name: string, unique identifier specifying the data set
      filenames: list of strings; each string is a path to an image file
      texts: list of strings; each string is human readable, e.g. 'dog'
      labels: list of integer; each integer identifies the ground truth
      num_shards: integer number of shards for this data set.
    """
    assert len(patch_paths) == len(patch_names)

    output_dir = os.path.join(FLAGS.output_directory, wsi_filename)
    print(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)

    sys.stdout.flush()

    # Create a generic tf-based utility for converting all image coding.
    coder = ImageCoder()
    output_filename = '%s-patches-%s' % (name, wsi_filename)
    output_file = os.path.join(output_dir, output_filename)
    writer = tf.python_io.TFRecordWriter(output_file)

    counter = 0
    start_time = time.time()
    for patch_path, patch_name in zip(patch_paths, patch_names):
        image_buffer, height, width = _process_image(patch_path, coder)

        example = _convert_to_example(image_buffer, patch_name)
        writer.write(example.SerializeToString())
        counter += 1

        if not counter % 1000:
            duration = time.time() - start_time
            print('Thread[%d]: %d secs - Processed %d of %d images.' %
                  (thread_index, math.ceil(duration), counter, len(patch_paths)))
            sys.stdout.flush()
            start_time = time.time()

    writer.close()

    print('Thread[%d]: %s: Finished writing all %d images in data set.' %
          (thread_index, datetime.now(), len(patch_paths)))
    sys.stdout.flush()


def _find_patches(thread_index, data_dir):
    """Build a list of all images files and labels in the data set.

    Args:
      data_dir: string, path to the root directory of images.(patches)

    """
    print('Thread[%d]: Determining list of file paths and names from %s.' % (thread_index, data_dir))

    file_names = os.listdir(data_dir)
    file_paths = glob.glob(data_dir + '/*')

    return file_paths, file_names


def _process_dataset(thread_index, name, directory, wsi_filename):
    """Process a complete data set and save it as a TFRecord.

    Args:
      name: string, unique identifier specifying the data set.
      directory: string, root path to the data set.
      num_shards: integer number of shards for this data set.
    """
    patch_paths, patch_names = _find_patches(thread_index, directory)
    _process_patches(thread_index, name, patch_paths, patch_names, wsi_filename)


def build_tf_records_split(thread_index, raw_patches_file_names):
    for wsi_filename in raw_patches_file_names:
        print('Thread[%d]: building tf_records for: %s' % (thread_index, wsi_filename))
        raw_patches_dir = os.path.join(utils.HEAT_MAP_RAW_PATCHES_DIR, wsi_filename)
        assert os.path.exists(raw_patches_dir), 'directory %s does not exist' % raw_patches_dir
        _process_dataset(thread_index, 'heatmap', raw_patches_dir, wsi_filename)


def main(unused_argv):
    assert not FLAGS.num_shards % FLAGS.num_threads, (
        'Please make the FLAGS.num_threads commensurate with FLAGS.num_shards')
    print('Saving results to %s' % FLAGS.output_directory)

    raw_patches_file_names = sorted(os.listdir(utils.HEAT_MAP_RAW_PATCHES_DIR))
    print(raw_patches_file_names)
    # raw_patches_file_names = raw_patches_file_names[8:9]

    assert len(raw_patches_file_names) % FLAGS.num_threads == 0, 'len(raw_patches_file_names) must be divisible by ' \
                                                                 'FLAGS.num_threads'

    items_per_split = int(len(raw_patches_file_names) / FLAGS.num_threads)
    raw_patches_file_names_splits = []

    for i in range(FLAGS.num_threads):
        raw_patches_file_names_splits.append(raw_patches_file_names[i * items_per_split: (i+1) * items_per_split])

    # coord = tf.train.Coordinator()

    # threads = []
    # for thread_index in range(FLAGS.num_threads):
    #     args = (thread_index, raw_patches_file_names_splits[thread_index])
    #     t = threading.Thread(target=build_tf_records_split, args=args)
    #     t.start()
    #     threads.append(t)

    # # Wait for all the threads to terminate.
    # coord.join(threads)
    # sys.stdout.flush()

    build_tf_records_split(0,raw_patches_file_names_splits[0])
    sys.stdout.flush()


if __name__ == '__main__':
    '''
    Convert extracted patches to tf-records
    '''
    tf.app.run()

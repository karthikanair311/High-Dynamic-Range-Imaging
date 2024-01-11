import tensorflow as tf
import math
import numpy as np
from config import Configuration
autotune = tf.data.experimental.AUTOTUNE
cfg = Configuration()

class Dataset():
    def __init__(self):
        self.train_gt_files = sorted(tf.io.gfile.glob(cfg.train_gt_path))
        self.train_gt_ar_files = sorted(tf.io.gfile.glob(cfg.train_gt_alignratio_path))
        self.train_medium_input_files = sorted(tf.io.gfile.glob(cfg.train_medium_input_path))
        self.train_long_input_files = sorted(tf.io.gfile.glob(cfg.train_long_input_path))
        self.train_short_input_files = sorted(tf.io.gfile.glob(cfg.train_short_input_path))
        self.num_train_imgs = len(self.train_medium_input_files)
        self.size_permutation_range = self.permute_resolutions()
        self.num_train_batches = math.ceil(self.num_train_imgs/cfg.train_batch_size)

        self.val_gt_files = sorted(tf.io.gfile.glob(cfg.val_gt_path))
        self.val_gt_ar_files = sorted(tf.io.gfile.glob(cfg.val_gt_alignratio_path))
        self.val_medium_input_files = sorted(tf.io.gfile.glob(cfg.val_medium_input_path))
        self.val_long_input_files = sorted(tf.io.gfile.glob(cfg.val_long_input_path))
        self.val_short_input_files = sorted(tf.io.gfile.glob(cfg.val_short_input_path))
        self.num_val_images = len(self.val_medium_input_files)

        self.train_ds = self.get_train_data()
        self.val_ds = self.get_val_data()



    def permute_resolutions(self):
        min_power = math.log2(cfg.min_train_res)
        max_power = math.log2((cfg.train_img_shape[0]*cfg.train_img_shape[1])//cfg.min_train_res)
        return [int(min_power),int(max_power)]

    def read_align_ratio(self, filename):
        align_ratio = np.load(filename.numpy()).astype(np.float32)
        return align_ratio

    def read_files(self, medium_input_img_path, long_input_img_path, short_input_img_path, target_img_path, gt_ds_alignratio):
        medium_input_img = tf.io.read_file(medium_input_img_path)
        medium_input_img = tf.image.decode_image(medium_input_img, dtype=tf.dtypes.uint8)

        long_input_img = tf.io.read_file(long_input_img_path)
        long_input_img = tf.image.decode_image(long_input_img, dtype=tf.dtypes.uint8)

        short_input_img = tf.io.read_file(short_input_img_path)
        short_input_img = tf.image.decode_image(short_input_img, dtype=tf.dtypes.uint8)

        target_img = tf.io.read_file(target_img_path)
        target_img = tf.image.decode_image(target_img, dtype=tf.dtypes.uint16)
        return medium_input_img, long_input_img, short_input_img, target_img, gt_ds_alignratio


    def create_pair(self, medium_input_img, long_input_img, short_input_img, target_img, gt_ds_alignratio):
        medium_input_img = tf.cast(medium_input_img, tf.dtypes.float32)
        medium_input_img = medium_input_img/255.0

        long_input_img = tf.cast(long_input_img, tf.dtypes.float32)
        long_input_img = long_input_img/255.0

        short_input_img = tf.cast(short_input_img, tf.dtypes.float32)
        short_input_img = short_input_img/255.0

        target_img = tf.cast(target_img, tf.dtypes.float32)
        target_img = target_img/gt_ds_alignratio[0]
        return tf.concat([medium_input_img, long_input_img, short_input_img, target_img], axis=-1)

    def create_train_crop(self, img_pair):
        img_patch = tf.image.random_crop(img_pair, [cfg.train_img_shape[0], cfg.train_img_shape[1], cfg.train_img_shape[-1]*4])
        return img_patch

    def create_val_crop(self, img_pair):
        img_patch = tf.image.random_crop(img_pair, [cfg.val_img_shape[0], cfg.val_img_shape[1], cfg.val_img_shape[-1]*4])
        return img_patch

    def split_train_pair(self, image_pair):
        return image_pair[:,:,:cfg.train_img_shape[-1]], image_pair[:,:,cfg.train_img_shape[-1]:cfg.train_img_shape[-1]*2], image_pair[:,:,cfg.train_img_shape[-1]*2:cfg.train_img_shape[-1]*3], image_pair[:,:,cfg.train_img_shape[-1]*3:]

    def split_val_pair(self, image_pair):
        return image_pair[:,:,:cfg.val_img_shape[-1]], image_pair[:,:,cfg.val_img_shape[-1]:cfg.val_img_shape[-1]*2], image_pair[:,:,cfg.val_img_shape[-1]*2:cfg.val_img_shape[-1]*3], image_pair[:,:,cfg.val_img_shape[-1]*3:]

    def train_augmentation(self, img_pair):
        img_pair = tf.image.random_flip_up_down(img_pair)
        img_pair = tf.image.random_flip_left_right(img_pair)
        img_pair = tf.image.rot90(img_pair, k=tf.random.uniform([], maxval=5, dtype=tf.int32))
        return img_pair

    def val_augmentation(self, img_pair):
        imgs_ud_flip = tf.image.flip_up_down(img_pair)
        imgs_lr_flip = tf.image.flip_left_right(img_pair)
        # img_pair_3 = tf.image.rot90(img_pair, k=tf.random.uniform([], maxval=5, dtype=tf.int32))
        img_pair = tf.concat([img_pair, imgs_ud_flip, imgs_lr_flip], axis=0)
        return img_pair

    def get_train_data(self):
        medium_input_ds = tf.data.Dataset.from_tensor_slices(self.train_medium_input_files)
        long_input_ds = tf.data.Dataset.from_tensor_slices(self.train_long_input_files)
        short_input_ds = tf.data.Dataset.from_tensor_slices(self.train_short_input_files)
        gt_ds = tf.data.Dataset.from_tensor_slices(self.train_gt_files)
        gt_ds_alignratio = tf.data.Dataset.from_tensor_slices(self.train_gt_ar_files)

        gt_ds_alignratio = gt_ds_alignratio.map(lambda fileName: tuple(tf.py_function(self.read_align_ratio, [fileName], [tf.float32])))
        ds = tf.data.Dataset.zip((medium_input_ds, long_input_ds, short_input_ds, gt_ds, gt_ds_alignratio))
        ds = ds.shuffle(buffer_size=1000, reshuffle_each_iteration=True)
        ds = ds.map(self.read_files, num_parallel_calls=autotune)
        # ds = ds.cache()
        ds = ds.shuffle(buffer_size=50, reshuffle_each_iteration=True)
        ds = ds.map(self.create_pair, num_parallel_calls=autotune)
        ds = ds.map(self.create_train_crop, num_parallel_calls=autotune)
        if cfg.train_augmentation:
            ds = ds.map(self.train_augmentation, num_parallel_calls=autotune)
        ds = ds.map(self.split_train_pair, num_parallel_calls=autotune)
        ds = ds.batch(cfg.train_batch_size, drop_remainder=False)
        ds = ds.prefetch(buffer_size=autotune)
        return ds

    def get_val_data(self):
        medium_input_ds = tf.data.Dataset.from_tensor_slices(self.val_medium_input_files)
        long_input_ds = tf.data.Dataset.from_tensor_slices(self.val_long_input_files)
        short_input_ds = tf.data.Dataset.from_tensor_slices(self.val_short_input_files)
        gt_ds = tf.data.Dataset.from_tensor_slices(self.val_gt_files)
        gt_ds_alignratio = tf.data.Dataset.from_tensor_slices(self.val_gt_ar_files)

        gt_ds_alignratio = gt_ds_alignratio.map(lambda fileName: tuple(tf.py_function(self.read_align_ratio, [fileName], [tf.float32])))
        ds = tf.data.Dataset.zip((medium_input_ds, long_input_ds, short_input_ds, gt_ds, gt_ds_alignratio))
        # ds = ds.shuffle(buffer_size=20, reshuffle_each_iteration=True)
        ds = ds.map(self.read_files, num_parallel_calls=autotune)
        # ds = ds.cache()
        ds = ds.map(self.create_pair, num_parallel_calls=autotune)
        ds = ds.map(self.create_val_crop, num_parallel_calls=autotune)
        if cfg.val_augmentation:
            ds = ds.map(self.val_augmentation, num_parallel_calls=autotune)
            # ds = ds.unbatch()
            # ds = ds.shuffle(buffer_size=20, reshuffle_each_iteration=True)
            # ds = ds.batch(cfg.val_batch_size)
        ds = ds.map(self.split_val_pair, num_parallel_calls=autotune)
        ds = ds.batch(cfg.val_batch_size, drop_remainder=False)
        # ds = ds.prefetch(buffer_size=autotune)
        return ds

import os
from shutil import copytree
from datetime import datetime

class Configuration():
    def __init__(self):


        # dataset config
        self.train_gt_path = 'train_dataset/*_gt.png'
        self.train_gt_alignratio_path = 'train_dataset/*_alignratio.npy'
        self.train_medium_input_path = 'train_dataset/*_medium.png'
        self.train_long_input_path = 'train_dataset/*_long.png'
        self.train_short_input_path = 'train_dataset/*_short.png'
        self.train_batch_size = 4
        self.train_img_shape = [128,128,3]
        self.min_train_res = 64
        self.train_augmentation = True

        self.val_gt_path = 'val_dataset/*_gt.png'
        self.val_gt_alignratio_path = 'val_dataset/*_alignratio.npy'
        self.val_medium_input_path = 'val_dataset/*_medium.png'
        self.val_long_input_path = 'val_dataset/*_long.png'
        self.val_short_input_path = 'val_dataset/*_short.png'
        self.val_batch_size = 2
        self.val_img_shape = [128,128,3]
        self.val_augmentation = False


        # # training config
        self.ckpt_dir = None # assign None if starting from scratch
        self.train_mode = ['best','last'][1]
        self.n_epochs = 10 #10000
        self.lr_boundaries = [9500,19000]
        self.lr_values= [2e-4, 1e-4, 1e-6]
        self.weight_mutone_loss = 0.9
        self.weight_cr_loss = 0.1

        #visualization config
        self.display_frequency = 5
        self.display_samples = 2
        self.log_dir = os.path.join('logs', str(datetime.now().strftime("%d%m%Y-%H%M%S")))

        # Parameters
        self.mu = 5000.0
        self.gamma = 2.24
        self.noFeat = 32

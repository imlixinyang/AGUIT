"""
Training codes for Attribute Guided Unpaired Image-to-image Translation.
Author: Xinyang Li (imlixinyang@gmail.com)
"""
"""
Original Copyright:
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from utils import get_all_data_loaders, prepare_sub_folder, write_loss, get_config, write_2images
import argparse
from trainer import AGUIT_Trainer
import torch
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import os
import sys
import tensorboardX
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/face.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument("--resume", action="store_true")
opts = parser.parse_args()


# Load experiment setting
config = get_config(opts.config)
max_iter = config['max_iter']
display_lize = config['display_size']
config['vgg_model_path'] = opts.output_path

# Setup model and data loader

trainer = AGUIT_Trainer(config)

trainer.cuda()
# load labeled and unlabeled dataloaders
train_loader_l, train_loader_u = get_all_data_loaders(config)

test_display_images_l = [torch.stack([train_loader_l.dataset[i][0] for i in range(display_lize)]).cuda(),
                         torch.stack([train_loader_l.dataset[i][1] for i in range(display_lize)]).cuda()]

# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0]
train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder

# Start training
iterations = trainer.resume(checkpoint_directory, hyperparameters=config) if opts.resume else 0
epoch = 0
import time
start = time.time()
while True:
    for it, (list_l, images_u) in enumerate(zip(train_loader_l, train_loader_u)):
        trainer.update_learning_rate()
        images_l, labels_l = list_l
        images_l, labels_l, images_u = images_l.cuda().detach(), labels_l.cuda().detach(), images_u.cuda().detach()
        # Avoid different number of images
        if images_l.size(0) > images_u.size(0):
            images_l, labels_l = images_l[0:images_u.size(0)], labels_l[0:images_u.size(0)]
        else:
            images_u = images_u[0:images_l.size(0)]
        # Model training
        loss_dis = trainer.dis_update(images_l, images_u, labels_l, config)
        loss_gen = trainer.gen_update(images_l, images_u, labels_l, config)
        torch.cuda.synchronize()

        # Log
        if (iterations + 1) % config['log_iter'] == 0:
            write_loss(iterations, trainer, train_writer)
            now = time.time()
            print(
                "[epoch:{:02d}#{:05d}|{:d}]genLoss:{:5.2f}, "
                "disLoss:{:5.2f}, with {:5.2f}s"
                    .format(epoch, iterations + 1, max_iter, loss_gen, loss_dis, now-start))
            start = now

        # Write images
        if (iterations + 1) % config['image_save_iter'] == 0:
            with torch.no_grad():
                test_image_outputs = trainer.sample(*test_display_images_l)
            write_2images(test_image_outputs, display_lize, image_directory, 'test_%08d' % (iterations + 1))

        # Save network weights
        if (iterations + 1) % config['snapshot_save_iter'] == 0:
            trainer.save(checkpoint_directory, iterations)

        iterations += 1
        if iterations >= max_iter:
            sys.exit('Finish training')

    epoch += 1


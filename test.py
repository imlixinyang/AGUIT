"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from utils import get_config, pytorch03_to_pytorch04
from trainer import AGUIT_Trainer
import argparse
from torch.autograd import Variable
import torchvision.utils as vutils
import sys
import torch
import os
from torchvision import transforms
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help="net configuration")
parser.add_argument('--input', type=str, help="input image path")
parser.add_argument('--output_folder', type=str, help="output image path")
parser.add_argument('--checkpoint', type=str, help="checkpoint of autoencoders")
parser.add_argument('--seed', type=int, default=10, help="random seed")
opts = parser.parse_args()

torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)
if not os.path.exists(opts.output_folder):
    os.makedirs(opts.output_folder)

# Load experiment setting
config = get_config(opts.config)
# Setup model and data loader

noise_dim = config['gen']['noise_dim']
attr_dim = len(config['gen']['selected_attrs'])
trainer = AGUIT_Trainer(config)
state_dict = torch.load(opts.checkpoint)
trainer.gen.load_state_dict(state_dict['gen'])

trainer.cuda()
trainer.eval()
encode = trainer.gen.encode
decode = trainer.gen.decode


new_size = config['new_size']
transform = transforms.Compose([transforms.Resize(new_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
def read_image(image_path):
    return Variable(transform(Image.open(opts.input).convert('RGB')).unsqueeze(0).cuda())

def write_image(x, path):
    outputs = (x + 1) / 2.
    path = os.path.join(opts.output_folder, path)
    vutils.save_image(outputs.data, path, padding=0, normalize=True)

with torch.no_grad():
    transform = transforms.Compose([transforms.Resize(new_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    image = read_image(opts.input)

    c, s = encode(image)
    x_t = decode(c, s)

    write_image(x_t, 'recon.jpg')

    """
    selection 1: multi-domain translation
    """
    s_t = s.clone()
    s_t[:, noise_dim + 1] = - s_t[:, noise_dim + 1]
    s_t[:, noise_dim + 4] = - s_t[:, noise_dim + 4]
    x_t = decode(c, s_t)

    write_image(x_t, 'multi-domain.jpg')

    """
    selection 2: multi-modal translation
    """
    s_t[:, noise_dim + 5] = - s_t[:, noise_dim + 5]
    s_t[:, :noise_dim] = torch.randn_like(s_t[:, :noise_dim])
    x_t = decode(c, s_t)

    write_image(x_t, 'multi-modal.jpg')

    """
    selection 3: diy your own translation
    """
    # DIY your own test code here
    pass




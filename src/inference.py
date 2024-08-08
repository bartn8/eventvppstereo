import os
import argparse

import torch
import numpy as np
import random

from manager import DLManager

import re

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, required=True)
parser.add_argument('--checkpoint_path', type=str, required=True)
parser.add_argument('--save_root', type=str, required=True)
parser.add_argument('--verbose', action='store_true', default=False)
parser.add_argument('--dataset', type=str, default='dsec', choices=['dsec', 'm3ed'])

parser.add_argument('--crop_height', type=int, default=0, help="""Crop the image. Must be divisible by 48""")
parser.add_argument('--crop_width', type=int, default=0, help="""Crop the image. Must be divisible by 48""")
parser.add_argument('--resize_height', type=int, default=0, help="""Resize image after crop it. Must be divisible by 48""")
parser.add_argument('--resize_width', type=int, default=0, help="""Resize image after crop it. Must be divisible by 48""")

parser.add_argument('--guide_method', nargs='+', default=['none'],
                     help="""Choose one or more guiding methods. \
                          'none': do not use guiding methods; \
                          'guided': Guided Stereo Matching; \
                          'bth': Back-in-Time Hallucination; \
                          'vsh': Virtual Stack Hallucination; \
                          'concat': Concat Lidar projection into input """)

parser.add_argument('--guideperc', type=float, default=0.15, help="""Percentage of hints sampled from GT if RAW is not available.""")
parser.add_argument('--randomperc', dest='norandomperc', action='store_false', default=True, help="""Enable random percentage of sampling.""")
parser.add_argument('--dilatehints', type=int, default=1, help='Dilate sparse hints using a square window. Default 1 (1x1)')

parser.add_argument('--vsh_patch_size', type=int, default=1, help="""VSH Patch size (default: 1)""")
parser.add_argument('--vsh_uniform_patch', action='store_true', help="""Use VSH uniform patch""")
parser.add_argument('--vsh_maskocc', action='store_true', help="""Use VSH mask occlusion""")
parser.add_argument('--vsh_splatting', action='store_true', help="""Use VSH subpixel splatting""")
parser.add_argument('--vsh_method', type=str, default='rnd', choices=['rnd',], help="""VSH Method""")
parser.add_argument('--vsh_filling', action='store_true', help="""Use VSH bilateral filling""")
parser.add_argument('--vsh_blending', type=float, help="""Set VSH alpha-blending""")

parser.add_argument('--bth_method', type=str, default='h-voxelgrid', choices=['h-naive', 'h-hist', 'h-voxelgrid', 'h-mdes', 'h-tore', 'h-timesurface'], help="""Choose a hallucination method based on event representation.""")
parser.add_argument('--bth_patch_size', type=int, default=1, help="""BTH Patch size (default: 1)""")
parser.add_argument('--bth_uniform_patch', action='store_true', help="""Use BTH uniform patch""")
parser.add_argument('--bth_uniform_polarities', action='store_true', help="""Use BTH uniform polarities""")
parser.add_argument('--bth_maskocc', action='store_true', help="""Use BTH mask occlusion""")
parser.add_argument('--bth_splatting', type=str, choices=['none', 'spatial', 'temporal'], help="""Use BTH subpixel splatting""")
parser.add_argument('--bth_n_bins', type=int, default=12, help="""Insert number of stack of downstream event representation""")
parser.add_argument('--bth_n_events', type=str, default='1', help="""Choose the number of fictitious events per pixel. Write a single number (e.g., '1') or a random range (e.g., '1:10') low and max included.""")
parser.add_argument('--bth_injection_timestamp', type=str, default='max', choices=['min', 'max', 'rnd'], help="""Choose injection timestamp. Valid only for 'h-naive'""")

parser.add_argument('--guided_k', type=float, default=10, help="""Guided Gaussian Amplitude (default: 10)""")
parser.add_argument('--guided_c', type=float, default=0.1, help="""Guided Gaussian Width (default: 0.1)""")

parser.add_argument('--raw_frameratio', type=int, default=1, help="""Set raw disparity framerate w.r.t. disparity computation. Default 1/1""")
parser.add_argument('--rnd_sampling', action='store_true', help="""Use random sampling instead of raw disparity""")
parser.add_argument('--raw_mae_threshold', type=float, default=0.1, help="""Set raw lidar MAE depth threshold to filter out raw frames with higher error""")

parser.add_argument('--render', action='store_true')
parser.add_argument('--save_predictions', action='store_true')
parser.add_argument('--savedir', type=str, default='toy_results')

parser.add_argument('--m3ed_raw_offset_us', type=int, default=0, help="""Set raw disparity negative offset w.r.t. disparity computation timestamp.""")
parser.add_argument('--m3ed_max_disparity_threshold', type=int, default=384, help="""Set M3ED max raw disparity threshold to filter out raw frames with higher disparity values""")
parser.add_argument('--m3ed_no_skip_frames', dest='m3ed_skip_frames', action='store_false', default=True, help="""Skip starting and ending frames (usually camera is static there).""")

parser.add_argument('--num_events', type=int, default=0)
parser.add_argument('--use_preproc_image', action='store_true')

parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--seed', type=int, default=42)

parser.add_argument('--csv_file', type=str, default=None, help="""Write results and parameters into a CSV file""")
parser.add_argument('--split', type=str, default='validation', choices=['validation', 'search', 'render', 'test'], help="""Choose split ('validation' or 'search')""")

parser.add_argument('--test_batch_size', type=int, default=1)
parser.add_argument('--seq_size', type=int, default=0)

args = parser.parse_args()

args.is_distributed = False
args.is_master = True
args.world_size = 1
args.local_rank = 0

args.guide_method = list(dict.fromkeys(args.guide_method))

for guide_method in args.guide_method:
    assert guide_method in ['none', 'guided', 'bth', 'vsh', 'concat']

assert re.match("^\d+(:\d+)?$", args.bth_n_events)

assert len(args.guide_method) > 0
assert len(args.guide_method) == 1 or 'none' not in args.guide_method, "Exclusive OR between 'none' and others guide methods"

assert os.path.isdir(args.data_root)

def set_global_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

# Set a specific seed, for example, 42
set_global_seed(args.seed)

exp_manager = DLManager(args)
exp_manager.load(args.checkpoint_path, args.dataset, args.crop_height, args.crop_width, args.num_events, args.use_preproc_image, args.test_batch_size, args.split)

exp_manager.test()

import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from .concentration import ConcentrationNet
from .baseline import StereoMatchingNetwork
from .. import models

import torch
import numpy as np

class EventStereoMatchingNetwork(nn.Module):
    def __init__(self, 
                 backbone=None,
                 skip_concentration_net=None,
                 concentration_net=None,
                 disparity_estimator=None):
        super(EventStereoMatchingNetwork, self).__init__()

        self.skip_concentration_net = skip_concentration_net

        if not skip_concentration_net:
            self.concentration_net = ConcentrationNet(**concentration_net.PARAMS)

        if backbone is not None:
            self.stereo_matching_net = getattr(models, backbone)(**disparity_estimator.PARAMS)
        else:
            self.stereo_matching_net = StereoMatchingNetwork(**disparity_estimator.PARAMS)
        
    def forward(self, left_event, right_event, gt_disparity=None, displ=None, guide=None, validguide=None,guide_args=None):
       
        event_stack = {
            'l': left_event.clone(),
            'r': right_event.clone(),
        }

        concentrated_event_stack = {}
        for loc in ['l', 'r']:
            event_stack[loc] = rearrange(event_stack[loc], 'b c h w t s -> b (c s t) h w')
            concentrated_event_stack[loc] = self.concentration_net(event_stack[loc]) if not self.skip_concentration_net else event_stack[loc].contiguous()

        pred_disparity_pyramid = self.stereo_matching_net(
            concentrated_event_stack['l'],
            concentrated_event_stack['r'], displ=displ, guide=guide, validguide=validguide, guide_args=guide_args
        )

        loss_disp = None
        if gt_disparity is not None:
            loss_disp = self.stereo_matching_net._cal_loss(pred_disparity_pyramid, gt_disparity)

        return pred_disparity_pyramid[-1], loss_disp

    def get_params_group(self, learning_rate):
        def filter_specific_params(kv):
            specific_layer_name = ['offset_conv.weight', 'offset_conv.bias']
            for name in specific_layer_name:
                if name in kv[0]:
                    return True
            return False

        def filter_base_params(kv):
            specific_layer_name = ['offset_conv.weight', 'offset_conv.bias']
            for name in specific_layer_name:
                if name in kv[0]:
                    return False
            return True

        specific_params = list(filter(filter_specific_params,
                                      self.named_parameters()))
        base_params = list(filter(filter_base_params,
                                  self.named_parameters()))

        specific_params = [kv[1] for kv in specific_params]  # kv is a tuple (key, value)
        base_params = [kv[1] for kv in base_params]

        specific_lr = learning_rate * 0.1
        params_group = [
            {'params': base_params, 'lr': learning_rate},
            {'params': specific_params, 'lr': specific_lr},
        ]

        return params_group

import torch
import torch.nn as nn
import torch.nn.functional as F

from .refinement import StereoDRNetRefinement

from .feature_extractor import FeatureExtractor
from .cost import CostVolumePyramid
from .aggregation import AdaptiveAggregation
from .estimation import DisparityEstimationPyramid
from .SparseConvNet import *
import numpy as np

from numba import njit

@njit
def project_lidar_to_right(displ_n):

    dispr_n = np.zeros_like(displ_n).astype(np.float32)
    B, C, H, W = displ_n.shape
    for b in range(B):
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    new_x = int(w - displ_n[b,c,h,w])
                    if new_x > 0:
                        dispr_n[b,c,h,new_x] = displ_n[b,c,h,w]
    return dispr_n


class StereoMatchingNetwork(nn.Module):
    def __init__(self, max_disp,
                 in_channels=3,
                 num_downsample=2,
                 no_mdconv=False,
                 feature_similarity='correlation',
                 num_scales=3,
                 num_fusions=6,
                 deformable_groups=2,
                 mdconv_dilation=2,
                 no_intermediate_supervision=False,
                 num_stage_blocks=1,
                 num_deform_blocks=3,
                 refine_channels=None, lidarinput=False):
        super(StereoMatchingNetwork, self).__init__()

        refine_channels = in_channels if refine_channels is None else refine_channels
        self.num_downsample = num_downsample
        self.num_scales = num_scales

        # Feature extractor
        self.feature_extractor = FeatureExtractor(in_channels=in_channels)

        self.lidarinput = lidarinput

        if self.lidarinput:
            print('Lidarstereonet-like!')
            self.feature_disp_pre = SparseConvNet()

        max_disp = max_disp // 3

        # Cost volume construction
        self.cost_volume_constructor = CostVolumePyramid(max_disp, feature_similarity=feature_similarity)

        # Cost aggregation
        self.aggregation = AdaptiveAggregation(max_disp=max_disp,
                                               num_scales=num_scales,
                                               num_fusions=num_fusions,
                                               num_stage_blocks=num_stage_blocks,
                                               num_deform_blocks=num_deform_blocks,
                                               no_mdconv=no_mdconv,
                                               mdconv_dilation=mdconv_dilation,
                                               deformable_groups=deformable_groups,
                                               intermediate_supervision=not no_intermediate_supervision)

        # Disparity estimation
        self.disparity_estimation = DisparityEstimationPyramid(max_disp)

        # Refinement
        refine_module_list = nn.ModuleList()
        for i in range(num_downsample):
            refine_module_list.append(StereoDRNetRefinement(img_channels=refine_channels))

        self.refinement = refine_module_list

        self.criterion = nn.L1Loss(reduction='none')

    def disparity_refinement(self, left_img, right_img, disparity):
        disparity_pyramid = []
        for i in range(self.num_downsample):
            scale_factor = 1. / pow(2, self.num_downsample - i - 1)

            if scale_factor == 1.0:
                curr_left_img = left_img
                curr_right_img = right_img
            else:
                curr_left_img = F.interpolate(left_img,
                                                scale_factor=scale_factor,
                                                mode='bilinear', align_corners=False)
                curr_right_img = F.interpolate(right_img,
                                                scale_factor=scale_factor,
                                                mode='bilinear', align_corners=False)
            inputs = (disparity, curr_left_img, curr_right_img)
            disparity = self.refinement[i](*inputs)
            disparity_pyramid.append(disparity)  # [H/2, H]

        return disparity_pyramid

    def guide_cost_volume(self, cost, hints, validhints, k=10, c=0.1): 
        GAUSSIAN_HEIGHT = k
        GAUSSIAN_WIDTH = c
        batch_size, num_disp, height, width = cost.shape
        SUBSAMPLE = hints.shape[-1] // cost.shape[-1]

        # image features are one fourth the original size: subsample the hints and divide them by four
        hints = hints.unsqueeze(1)
        hints = F.interpolate(hints, (height, width), mode='nearest')
        validhints = validhints.unsqueeze(1)
        validhints = F.interpolate(validhints, (height, width), mode='nearest')
        hints = hints*validhints / float(SUBSAMPLE)
        GAUSSIAN_WIDTH /= float(SUBSAMPLE)

        # add feature and disparity dimensions to hints and validhints
        # and repeat their values along those dimensions, to obtain the same size as cost
        hints = hints.expand(-1, num_disp, -1, -1)
        validhints = validhints.expand(-1, num_disp, -1, -1)
        
        # create a tensor of the same size as cost, with disparities
        # between 0 and num_disp-1 along the disparity dimension
        disparities = torch.linspace(start=0, end=num_disp - 1, steps=num_disp).cuda().unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(batch_size, -1, height, width)
        cost = cost * ((1 - validhints) + validhints * GAUSSIAN_HEIGHT * torch.exp(-(disparities - hints) ** 2 / (2 * GAUSSIAN_WIDTH ** 2)))
        return cost

    def forward(self, left_img, right_img, displ=None, guide=None, validguide=None, guide_args=None):
        left_feature = self.feature_extractor(left_img)
        right_feature = self.feature_extractor(right_img)

        if not self.lidarinput and displ is not None:
            raise Exception("concat guide strategy not supported. Retrain with lidarinput: True in config.yaml")
        
        if displ is not None:
            displ = displ.unsqueeze(1)
            dispr = torch.from_numpy(project_lidar_to_right(displ.cpu().numpy())).cuda()
            
            lidarleft_feature = self.feature_disp_pre(displ) 
            lidarright_feature = self.feature_disp_pre(dispr) 

            left_feature = [torch.cat((l,d), 1) for l,d in zip(left_feature,lidarleft_feature)]
            right_feature = [torch.cat((l,d), 1) for l,d in zip(right_feature,lidarright_feature)]

        cost_volume = self.cost_volume_constructor(left_feature, right_feature)

        if guide is not None: 
            validguide = validguide.float()
            guided_k = guide_args['guided_k']
            guided_c = guide_args['guided_c']
            cost_volume = [self.guide_cost_volume(cost=c, hints=guide, validhints=validguide, k=guided_k, c=guided_c) for c in cost_volume]

        aggregation = self.aggregation(cost_volume)
        disparity_pyramid = self.disparity_estimation(aggregation)
        disparity_pyramid += self.disparity_refinement(left_img, right_img, disparity_pyramid[-1])

        return disparity_pyramid


    def _cal_loss(self, pred_disparity_pyramid, gt_disparity):
        pyramid_weight = [1 / 3, 2 / 3, 1.0, 1.0, 1.0]

        loss = 0.0
        mask = gt_disparity > 0
        for idx in range(len(pyramid_weight)):
            pred_disp = pred_disparity_pyramid[idx]
            weight = pyramid_weight[idx]

            if pred_disp.size(-1) != gt_disparity.size(-1):
                pred_disp = pred_disp.unsqueeze(1)
                pred_disp = F.interpolate(pred_disp, size=(gt_disparity.size(-2), gt_disparity.size(-1)),
                                          mode='bilinear', align_corners=False) * (
                                    gt_disparity.size(-1) / pred_disp.size(-1))
                pred_disp = pred_disp.squeeze(1)

            cur_loss = self.criterion(pred_disp[mask], gt_disparity[mask])
            loss += weight * cur_loss

        return loss
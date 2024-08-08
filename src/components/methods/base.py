import torch
import matplotlib.pyplot as plt
import cv2

from tqdm import tqdm
import numpy as np

from utils import visualizer

from .filter import occlusion_heuristic
from .vsh import vsh
from .utils import sample_hints

import numpy as np

import torch.nn.functional as F

import os

@torch.no_grad()
def test(model, data_loader, log_dict, stacking_method, args=None, seq_name=None, seq_log_dict = None):
    model.eval()
    pred_list = []

    guide_method = args.guide_method if args is not None else ['none']
    guideperc = args.guideperc if args is not None and 0 <= args.guideperc <= 1 else 0.15

    guide_args = {
        'guide_method': guide_method,
        'stacking_method': stacking_method,

        'guided_k': args.guided_k if args is not None else 10,
        'guided_c': args.guided_c if args is not None else 0.1,

        'vsh_patch_size': args.vsh_patch_size if args is not None else 1,
        'vsh_uniform_patch': args.vsh_uniform_patch if args is not None else False,
        'vsh_maskocc': args.vsh_maskocc if args is not None else False,
        'vsh_splatting': args.vsh_splatting if args is not None else False,
        'vsh_method': args.vsh_method if args is not None else 'rnd',
        'vsh_filling': args.vsh_filling if args is not None else False,
        'vsh_blending': args.vsh_blending if args is not None else 1.0
    }

    with tqdm(data_loader, dynamic_ncols=True) as loader:
        idx=0
        for batch_data in loader:
            batch_data = batch_to_cuda(batch_data)

            mask = batch_data['disparity'] > 0
            if not mask.any():
                continue
             
            if 'none' not in guide_method:
                if 'disparity_raw' in batch_data:
                    hints, validhints = batch_data['disparity_raw'], batch_data['disparity_raw']>0
                else:
                    hints, validhints = sample_hints(batch_data['disparity'], mask, guideperc)
                
                mydevice = hints.get_device()
            else:
                hints, validhints = None, None

            vanilla_left_event = batch_data['event']['left']
            vanilla_right_event = batch_data['event']['right']

            B, C, H, W, T, S = vanilla_left_event.shape

            event_stack_left = vanilla_left_event.permute(0,1,5,4,2,3).reshape(B,-1,H,W)
            event_stack_right = vanilla_right_event.permute(0,1,5,4,2,3).reshape(B,-1,H,W)

            resize_height = args.resize_height if args is not None and args.resize_height > 0 else None
            resize_width = args.resize_width if args is not None and args.resize_width > 0 else None

            if resize_height is not None and resize_width is not None and resize_height > 0 and resize_width > 0:            
                event_stack_left = F.interpolate(event_stack_left, (resize_height, resize_width), mode='nearest')
                event_stack_right = F.interpolate(event_stack_right, (resize_height, resize_width), mode='nearest')
                disparity = (F.interpolate(batch_data['disparity'].unsqueeze(1), (resize_height, resize_width), mode='nearest') * (resize_width / W)).squeeze(1)
                mask = disparity > 0

                if hints is not None:
                    hints = (F.interpolate(hints.unsqueeze(1), (resize_height, resize_width), mode='nearest') * (resize_width / W)).squeeze(1)
                    validhints = hints > 0
            else:
                resize_width = W
                resize_height = H
                disparity = batch_data['disparity']

            vanilla_left_event = event_stack_left.reshape(B,C,S,T,resize_height, resize_width).permute(0, 1, 4, 5, 3, 2)
            vanilla_right_event = event_stack_right.reshape(B,C,S,T,resize_height, resize_width).permute(0, 1, 4, 5, 3, 2)

            if 'vsh' in guide_method:
                tmp_stack_left = []
                tmp_stack_right = []
                
                for b in range(hints.shape[0]):
                    event_left = event_stack_left[b].permute(1,2,0).cpu().numpy()
                    event_right = event_stack_right[b].permute(1,2,0).cpu().numpy()
                    myhints = hints[b].cpu().numpy()
                    occ_mask = occlusion_heuristic(myhints) if guide_args['vsh_maskocc'] else None
                    vpp_left, vpp_right = vsh(event_left, event_right, myhints, occ_mask, guide_args)

                    tmp_stack_left.append(torch.from_numpy(vpp_left).permute(2,0,1).unsqueeze(0))
                    tmp_stack_right.append(torch.from_numpy(vpp_right).permute(2,0,1).unsqueeze(0))        

                tmp_stack_left = torch.cat(tmp_stack_left,0).to(mydevice)
                tmp_stack_right = torch.cat(tmp_stack_right,0).to(mydevice)

                vpp_left_event = tmp_stack_left.reshape(B,C,S,T,resize_height, resize_width).permute(0, 1, 4, 5, 3, 2)
                vpp_right_event = tmp_stack_right.reshape(B,C,S,T,resize_height, resize_width).permute(0, 1, 4, 5, 3, 2)
            else:
                vpp_left_event,vpp_right_event = vanilla_left_event,vanilla_right_event

            guide, validguide = None, None
            lidar, validlidar = None, None

            if 'guided' in guide_method:
                guide, validguide = hints, validhints
            if 'concat' in guide_method:
                lidar, validlidar = hints, validhints

            pred, _ = model(left_event=vpp_left_event,
                            right_event=vpp_right_event,
                            gt_disparity=disparity, displ=lidar, guide=guide, validguide=validguide, guide_args=guide_args)
            
            if args.render:
                width = data_loader.dataset.WIDTH
                height = data_loader.dataset.HEIGHT
                es_left = vanilla_left_event.permute(0,1,5,4,2,3).reshape(B,-1,resize_height, resize_width)[0,:,:height, :width].cpu()
                es_right = vanilla_right_event.permute(0,1,5,4,2,3).reshape(B,-1,resize_height, resize_width)[0,:,:height, :width].cpu()
                es_left, es_right = torch.mean(es_left, dim=0), torch.mean(es_right, dim=0)

                if 'vsh' in guide_method:
                    es_left_h = vpp_left_event.permute(0,1,5,4,2,3).reshape(B,-1,resize_height, resize_width)[0,:,:height, :width].cpu()
                    es_right_h = vpp_right_event.permute(0,1,5,4,2,3).reshape(B,-1,resize_height, resize_width)[0,:,:height, :width].cpu()
                    es_left_h, es_right_h = torch.mean(es_left_h, dim=0), torch.mean(es_right_h, dim=0)

                cur_pred = pred[0, :data_loader.dataset.HEIGHT, :data_loader.dataset.WIDTH].cpu()
                cur_gt = disparity[0, :height, :width].cpu()
                cur_errormap = visualizer.color_error_image_kitti(torch.abs(cur_gt-cur_pred).cpu().squeeze().numpy(), scale=2, mask=cur_gt>0, dilation=3)

                def calculate_error(pred, ground_truth, mask, n=1):
                    # pred, ground_truth, mask: (H, W)
                    pred, ground_truth = pred[mask], ground_truth[mask]
                    error = torch.abs(pred - ground_truth)
                    error_mask = error > n
                    error_mask = error_mask.to(torch.float)

                    final_error = error_mask.mean()

                    return final_error * 100.0

                print(f"{idx}) 1PE: {calculate_error(cur_pred,cur_gt,cur_gt>0,n=1)} %")

                kernel_7x7 = np.ones((7,7))

                plt.imsave('%s/%s_%05d_norm.jpg'%(args.savedir,seq_name,idx), cur_pred, cmap='magma')
                plt.imsave('%s/%s_%05d.jpg'%(args.savedir,seq_name,idx), torch.clip(cur_pred, 0, cur_gt.max()), cmap='magma', vmin=0, vmax=cur_gt.max())
                plt.imsave('%s/%s_%05d_gt.jpg'%(args.savedir,seq_name,idx), cv2.dilate(cur_gt.squeeze().cpu().numpy(), kernel_7x7), cmap='magma', vmin=0)
                cv2.imwrite('%s/%s_%05d_errormap.jpg'%(args.savedir,seq_name,idx), cur_errormap)
                
                if hints is not None:
                    plt.imsave('%s/%s_%05d_raw.jpg'%(args.savedir,seq_name,idx), cv2.dilate((torch.clip(hints, 0, cur_gt.max())).cpu().squeeze().numpy(), kernel_7x7), cmap='magma', vmin=0, vmax=cur_gt.max())
                
                plt.imsave('%s/%s_%05d_es_left.jpg'%(args.savedir,seq_name,idx), es_left/es_left.max(), cmap='magma')
                plt.imsave('%s/%s_%05d_es_right.jpg'%(args.savedir,seq_name,idx), es_right/es_right.max(), cmap='magma')

                if 'vsh' in guide_method:
                    plt.imsave('%s/%s_%05d_es_left_h.jpg'%(args.savedir,seq_name,idx), es_left_h/es_left_h.max(), cmap='magma')
                    plt.imsave('%s/%s_%05d_es_right_h.jpg'%(args.savedir,seq_name,idx), es_right_h/es_right_h.max(), cmap='magma')

            idx+=1

            if resize_height is not None and resize_width is not None and resize_height > 0 and resize_width > 0:
                original_height, original_width = batch_data['disparity'].shape[-2], batch_data['disparity'].shape[-1]
                if original_height != resize_height or original_width != resize_width:
                    pred = (F.interpolate(pred.unsqueeze(0), (original_height, original_width), mode='bilinear') * (original_width / resize_width)).squeeze(0)
                    disparity = batch_data['disparity']
                    mask = disparity > 0

            log_dict['EPE'].update(pred, disparity, mask)
            log_dict['1PE'].update(pred, disparity, mask)
            log_dict['2PE'].update(pred, disparity, mask)
            log_dict['3PE'].update(pred, disparity, mask)
            log_dict['RMSE'].update(pred, disparity, mask)

            if seq_log_dict is not None:
                seq_log_dict['EPE'].update(pred, disparity, mask)
                seq_log_dict['1PE'].update(pred, disparity, mask)
                seq_log_dict['2PE'].update(pred, disparity, mask)
                seq_log_dict['3PE'].update(pred, disparity, mask)
                seq_log_dict['RMSE'].update(pred, disparity, mask)

            loader.set_description('Sequence %s, 1PE: %.4f'%(seq_name, log_dict['1PE'].calculate_error(pred, disparity, mask).item()))

            #Save prediction in KITTI format
            if args.save_predictions:
                width = data_loader.dataset.WIDTH
                height = data_loader.dataset.HEIGHT
                cur_pred = pred[0, :height, :width].cpu().numpy()
                quantized_pred = np.clip(cur_pred*256.0, 0, 65535).astype(np.uint16)
                
                os.makedirs(os.path.join(args.savedir, seq_name), exist_ok=True)
                cv2.imwrite(os.path.join(args.savedir,seq_name,str(idx).zfill(6) + '.png'), quantized_pred)              
            
    return pred_list


def batch_to_cuda(batch_data):
    def _batch_to_cuda(batch_data):
        if isinstance(batch_data, dict):
            for key in batch_data.keys():
                batch_data[key] = _batch_to_cuda(batch_data[key])
        elif isinstance(batch_data, torch.Tensor):
            batch_data = batch_data.cuda()
        else:
            raise NotImplementedError

        return batch_data

    for domain in ['event']:
        if domain not in batch_data.keys():
            batch_data[domain] = {}
        for location in ['left', 'right']:
            if location in batch_data[domain].keys():
                batch_data[domain][location] = _batch_to_cuda(batch_data[domain][location])
            else:
                batch_data[domain][location] = None

    if 'disparity' in batch_data.keys() and batch_data['disparity'] is not None:
        batch_data['disparity'] = batch_data['disparity'].cuda()

    if 'disparity_raw' in batch_data.keys() and batch_data['disparity_raw'] is not None:
        batch_data['disparity_raw'] = batch_data['disparity_raw'].cuda()

    return batch_data

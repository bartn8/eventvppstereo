import os

import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from collections import OrderedDict
from utils.metric import AverageMeter, EndPointError, NPixelError, RootMeanSquareError, RelError, DeltaError

from components import models
from components import datasets
from components import methods

from utils.logger import ExpLogger, TimeCheck
from utils.metric import SummationMeter, Metric

from yacs.config import CfgNode as CN


class DLManager:
    def __init__(self, args, cfg=None):
        self.args = args
        self.cfg = cfg
        self.logger = ExpLogger(save_root=args.save_root) if args.is_master else None
        if self.cfg is not None:
            self.log_interval = self.cfg.LOG_INTERVAL

        if self.cfg is not None:
            self._init_from_cfg(cfg)

        self.current_epoch = 0

    def _init_from_cfg(self, cfg:CN, dataset_name = None, crop_height = None, crop_width = None, num_events = None, use_preprocessed_image = None, test_batch_size = None):
        assert cfg is not None
        self.cfg = cfg

        self.cfg.defrost()
        if dataset_name is not None:
            self.cfg.DATASET.TRAIN.NAME = dataset_name
        if crop_height is not None and crop_height > 0:
            self.cfg.DATASET.TEST.PARAMS.crop_height = crop_height
        if crop_width is not None and crop_width > 0:
            self.cfg.DATASET.TEST.PARAMS.crop_width = crop_width
        if num_events is not None and num_events > 0:
            self.cfg.DATASET.TRAIN.PARAMS.event_cfg.PARAMS.num_of_event = num_events
            self.cfg.DATASET.TEST.PARAMS.event_cfg.PARAMS.num_of_event = num_events
        if use_preprocessed_image is not None:
            self.cfg.DATASET.TRAIN.PARAMS.event_cfg.PARAMS.use_preprocessed_image = use_preprocessed_image
            self.cfg.DATASET.TEST.PARAMS.event_cfg.PARAMS.use_preprocessed_image = use_preprocessed_image
        if test_batch_size is not None:
            self.cfg.DATALOADER.TEST.PARAMS.batch_size = test_batch_size
        self.cfg.freeze()

        self.model = _prepare_model(self.cfg.MODEL,
                                    is_distributed=self.args.is_distributed,
                                    local_rank=self.args.local_rank if self.args.is_distributed else None)
        self.optimizer = _prepare_optimizer(self.cfg.OPTIMIZER, self.model)
        self.scheduler = _prepare_scheduler(self.cfg.SCHEDULER, self.optimizer)

        self.get_train_loader = getattr(datasets, self.cfg.DATASET.TRAIN.NAME).get_dataloader
        self.get_test_loader = getattr(datasets, self.cfg.DATASET.TRAIN.NAME).get_dataloader

        self.method = getattr(methods, self.cfg.METHOD)

    def test(self):
        if self.args.is_master:
            test_loader = self.get_test_loader(args=self.args,
                                               dataset_cfg=self.cfg.DATASET.TRAIN,
                                               dataloader_cfg=self.cfg.DATALOADER.TEST)

            self.logger.test()

            log_dict = OrderedDict([
                ('EPE', EndPointError(average_by='image', string_format='%5.2lf')),
                ('1PE', NPixelError(n=1, average_by='image', string_format='%5.2lf')),
                ('2PE', NPixelError(n=2, average_by='image', string_format='%5.2lf')),
                ('3PE', NPixelError(n=3, average_by='image', string_format='%5.2lf')),
                ('RMSE', RootMeanSquareError(average_by='image', string_format='%5.2lf')),
            ])

            for sequence_dataloader in test_loader:
                sequence_name = sequence_dataloader.dataset.sequence_name

                seq_log_dict = OrderedDict([
                    ('EPE', EndPointError(average_by='image', string_format='%5.2lf')),
                    ('1PE', NPixelError(n=1, average_by='image', string_format='%5.2lf')),
                    ('2PE', NPixelError(n=2, average_by='image', string_format='%5.2lf')),
                    ('3PE', NPixelError(n=3, average_by='image', string_format='%5.2lf')),
                    ('RMSE', RootMeanSquareError(average_by='image', string_format='%5.2lf')),
                ])

                self.method.test(model=self.model,
                                 stacking_method=self.cfg.DATASET.TEST.PARAMS.event_cfg.PARAMS.stack_method,
                                 data_loader=sequence_dataloader, log_dict=log_dict, args=self.args, seq_name=sequence_name, seq_log_dict=seq_log_dict)
                
                if self.args.verbose:
                    print(f"Sequence {sequence_name} metrics")
                    for k in seq_log_dict.keys():
                        print('%s: %.2f'%(k,seq_log_dict[k].value))

            if self.args.verbose:
                print("-"*40)

            for k in log_dict.keys():
                print('%s: %.2f'%(k,log_dict[k].value))

            csv_file = self.args.csv_file
            if csv_file is not None:
                args_dict = vars(self.args)
                write_header = not os.path.exists(csv_file)
                
                with open(csv_file, "a") as f:
                    dict_keys = list(args_dict.keys())
                    result_keys = list(log_dict.keys())
                    if write_header:
                        for key in dict_keys:
                            f.write(f"{key},")
                        
                        for key in result_keys[:-1]:
                            f.write(f"{key},")

                        f.write(f"{result_keys[-1]}\n")

                    for key in dict_keys:
                        f.write(f"{args_dict[key]},")
                    
                    for key in result_keys[:-1]:
                        f.write(f"{log_dict[key]},")

                    f.write(f"{log_dict[result_keys[-1]]}\n")

            
    def save(self, name):
        checkpoint = self._make_checkpoint()
        self.logger.save_checkpoint(checkpoint, name)

    def resume(self, name):
        checkpoint = self.logger.load_checkpoint(name)

        self.current_epoch = checkpoint['epoch']
        self.model.module.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])

    def load(self, name, dataset_name, crop_height, crop_width, num_events, use_preprocessed_image, test_batch_size, split='validation'):
        checkpoint = self.logger.load_checkpoint(name)
        self._init_from_cfg(checkpoint['cfg'], dataset_name, crop_height, crop_width, num_events, use_preprocessed_image, test_batch_size)

        self.cfg.defrost()
        self.cfg.DATASET.TRAIN.PARAMS.split = split
        self.cfg.DATASET.TRAIN.PARAMS.crop_height = self.cfg.DATASET.TEST.PARAMS.crop_height
        self.cfg.DATASET.TRAIN.PARAMS.crop_width = self.cfg.DATASET.TEST.PARAMS.crop_width
        self.cfg.freeze()

        self.model.module.load_state_dict(checkpoint['model'])

    def _make_checkpoint(self):
        checkpoint = {
            'epoch': self.current_epoch,
            'args': self.args,
            'cfg': self.cfg,
            'model': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }

        return checkpoint

    def _gather_log(self, log_dict):
        if log_dict is None:
            return None

        for key in log_dict.keys():
            if isinstance(log_dict[key], SummationMeter) or isinstance(log_dict[key], Metric):
                log_dict[key].all_gather(self.args.world_size)

        return log_dict

    def _log_after_epoch(self, epoch, time_checker, log_dict, part):
        # Calculate Time
        time_checker.update(epoch)

        # Log Time
        self.logger.write('Epoch: %d | time per epoch: %s | eta: %s' %
                          (epoch, time_checker.time_per_epoch, time_checker.eta))

        # Log Learning Process
        log = '%5s' % part
        for key in log_dict.keys():
            log += ' | %s: %s' % (key, str(log_dict[key]))
            if isinstance(log_dict[key], SummationMeter) or isinstance(log_dict[key], Metric):
                self.logger.add_scalar('%s/%s' % (part, key), log_dict[key].value, epoch)
            else:
                self.logger.add_scalar('%s/%s' % (part, key), log_dict[key], epoch)
        self.logger.write(log=log)

        # Make Checkpoint
        checkpoint = self._make_checkpoint()

        # Save Checkpoint
        self.logger.save_checkpoint(checkpoint, 'final.pth')
        if epoch % self.args.save_term == 0:
            self.logger.save_checkpoint(checkpoint, '%d.pth' % epoch)


def _prepare_model(model_cfg, is_distributed=False, local_rank=None):
    name = model_cfg.NAME
    parameters = model_cfg.PARAMS

    model = getattr(models, name)(**parameters)
    if is_distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    else:
        model = nn.DataParallel(model).cuda()

    return model


def _prepare_optimizer(optimizer_cfg, model):
    name = optimizer_cfg.NAME
    parameters = optimizer_cfg.PARAMS
    learning_rate = parameters.lr

    params_group = model.module.get_params_group(learning_rate)

    optimizer = getattr(optim, name)(params_group, **parameters)

    return optimizer


def _prepare_scheduler(scheduler_cfg, optimizer):
    name = scheduler_cfg.NAME
    parameters = scheduler_cfg.PARAMS

    if name == 'CosineAnnealingWarmupRestarts':
        from utils.scheduler import CosineAnnealingWarmupRestarts
        scheduler = CosineAnnealingWarmupRestarts(optimizer, **parameters)
    else:
        scheduler = getattr(optim.lr_scheduler, name)(optimizer, **parameters)

    return scheduler

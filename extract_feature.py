from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import numpy as np
import sys
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import os
import pickle


from reid import datasets
from reid import models
from reid.utils.data import transforms as T
from reid.utils.serialization import load_checkpoint
from reid.datasets import JsonDataset
from reid.feature_extraction import extract_cnn_feature


'''
This is the code for paper 'parameter-free spatial attention network for Person Re-Identification'
Our code is mainly based on PCB 
'''




def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True

    # Create data loaders
    if args.height is None or args.width is None:
        args.height, args.width = (144, 56) if args.arch == 'inception' else \
                                  (256, 128)


    # Create model
    model = models.create(args.arch, num_features=args.features,
                          dropout=args.dropout, num_classes=1648)

    # Load from checkpoint
    start_epoch = best_top1 = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model_dict = model.state_dict()
        checkpoint_load = {k: v for k, v in (checkpoint['state_dict']).items() if k in model_dict}
        model_dict.update(checkpoint_load)
        model.load_state_dict(model_dict)
#        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        best_top1 = checkpoint['best_top1']
        print("=> Start epoch {}  best top1 {:.1%}"
              .format(start_epoch, best_top1))

    model = nn.DataParallel(model).cuda()


    normalize_transform = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    transform = T.Compose([
        T.Resize((args.height, args.width)),
        T.ToTensor(),
        normalize_transform
    ])
    val_can_feats = {}
    if args.test_or_val == 'val':
        DATA_DIR = os.path.join(args.data_dir, 'val_bbox')
        gallery = JsonDataset(
            DATA_DIR, os.path.join(args.data_dir, 'val.json'), transform, state='can')
        output_name = 'val_features.pkl'
    elif args.test_or_val == 'test':
        DATA_DIR = os.path.join(args.data_dir, 'test_bbox')
        gallery = JsonDataset(
            DATA_DIR, os.path.join(args.data_dir, 'test.json'), transform, state='can')
        output_name = 'test_features.pkl'
    else:
        raise KeyError

    gallery_loader = torch.utils.data.DataLoader(gallery,
                                                 batch_size=200, shuffle=False,
                                                 num_workers=8, pin_memory=True)

    for idx, (data, target) in enumerate(gallery_loader):
        data = data.cuda()
        with torch.no_grad():
            output = extract_cnn_feature(model, data)
        output = output.view(output.shape[0], -1)
        feat = output.squeeze().cpu().detach().numpy()
        for idd, k in enumerate(target):
            val_can_feats[k] = np.float16(feat[idd])
        if idx % 20 == 0:
            print('extracting {}/{} batches'.format(idx, len(gallery_loader)))

    with open(os.path.join(args.logs_dir, output_name), 'wb') as f:
        pickle.dump(val_can_feats, f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Softmax loss classification")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='market',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--height', type=int, default=384,
                        help="input height, default: 256 for resnet*, "
                             "144 for inception")
    parser.add_argument('--width', type=int, default=128,
                        help="input width, default: 128 for resnet*, "
                             "56 for inception")
    parser.add_argument('--combine-trainval', action='store_true',
                        help="train and val sets together for training, "
                             "val set alone for validation")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--test_or_val', type=str, default='val')
    parser.add_argument('--features', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.1,
                        help="learning rate of new parameters, for pretrained "
                             "parameters it is 10 times smaller than this")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--step-size',type=int, default=40)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=1)
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    main(parser.parse_args())

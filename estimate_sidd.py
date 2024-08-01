import argparse
import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torchvision import models
from models.resnet_simclr import ResNetSimCLR
import data.dataset as datasets
import matplotlib.pyplot as plt
from tqdm import tqdm
from calibration import joint_params_dist

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
parser = argparse.ArgumentParser(description='PyTorch Noise Estimation')
parser.add_argument('--data_path', default='./dataset/SIDD_Medium_Raw/Data',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--camera', default='S6', type=str, choices=['S6', 'IP', 'GP', 'N6', 'G4'], help='choose camera type from SIDD dataset')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')


def main():
    args = parser.parse_args()
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    basedir = args.data_path
    camera = args.camera
    eval_dataset = datasets.SiddEvalDataset(datadir=basedir, camera=camera, size=None, amp_ratio=1, repeat=5)

    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim).to(args.device)
    checkpoint = torch.load('pretrained/noise_estimation_model.pth.tar', map_location='cuda')   # PgRB
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict, strict=False)

    with torch.no_grad():
        model.eval()
        tqdm_loader = tqdm(eval_loader)
        camera_params = []
        for i, dic in enumerate(tqdm_loader):
            images = dic['input']
            # 2B*C*H*W
            images = images.to(args.device)
            iso = dic['iso'][:, None, None, None].float().to(args.device)
            features, params, row_param, bias_param = model(images, iso)   # dim: 2B*128
            total_params = torch.cat([params, row_param, bias_param], dim=1)
            total_params = np.squeeze(total_params.detach().cpu().numpy())
            iso = iso.detach().cpu().numpy()[0, 0, 0]
            camera_params.append(np.concatenate((total_params, iso)))
    camera_params = np.array(camera_params)
    camera_params = camera_params[np.argsort(camera_params[:, -1])]
    np.save('./sidd_params/{}.npy'.format(camera), camera_params)
    joint_params = joint_params_dist(camera_params, camera=camera)
    params_path = './sidd_params/{}_params.npy'.format(camera)
    if not os.path.exists(params_path):
        print('save {}..'.format(params_path))
        np.save(params_path, joint_params)

if __name__ == "__main__":
    main()

""" import torch
import argparse
import os
import random
import torch.optim as optim
import numpy as np
import shutil
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from nt_xent_sup_sparse import NT_Xent_Sup
from dataset import KadisDataset
from model5 import CModel
import torchvision.transforms as transforms

def adjust_learning_rate(optimizer, epoch, lr_decay_epoch=8):
    # Decay learning rate by a factor every `lr_decay_epoch` epochs.
    if (epoch + 1) % lr_decay_epoch == 0:
        decay_rate = 0.9 ** (epoch // lr_decay_epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay_rate
    return optimizer

def train_one_epoch(dataloader_train, model, loss_fn, optimizer, scheduler, device):
    # 한 에폭 동안의 훈련 루프
    loss_epoch = []
    model.train()
    loop = tqdm(dataloader_train, total=len(dataloader_train), desc="Training")

    for batch_idx, (dist, label_dist) in enumerate(loop):
        dist = dist.to(device).float()
        label_dist = label_dist.to(device)

        optimizer.zero_grad()

        z1, z2, _ = model(dist, dist)

        # 손실 계산 및 출력
        print(f"z1: {z1}, z2: {z2}, label_dist: {label_dist}")

        loss = loss_fn(z1, z2, label_dist)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        # 손실 출력
        print(f"Batch {batch_idx} Loss: {loss.item()}")
        loss_epoch.append(loss.item())
        loop.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])

    return np.mean(loss_epoch)

def main(args):
    # 훈련 메인 함수
    # 경로 설정
    root_dir = 'C:/Users/IIPL02/Desktop/saTQA/SaTQA/IQA_dataset/kadis700k/kadis700k'
    csv_path = 'C:/Users/IIPL02/Desktop/saTQA/SaTQA/IQA_dataset/kadis700k/kadis_new_with_paths.csv'
    dist_img_dir = os.path.join(root_dir, 'dist_imgs')

    # CSV 파일 경로 출력 및 확인
    print(f"CSV 파일 경로: {csv_path}")
    if os.path.exists(csv_path):
        print("CSV 파일이 존재합니다.")
    else:
        print("CSV 파일을 찾을 수 없습니다.")
        return

    # 시드 설정
    if args.seed != 0:
        print(f'SEED = {args.seed}')
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.backends.cudnn.deterministic = True

    # 폴더 생성
    os.makedirs(args.sv_path, exist_ok=True)
    os.makedirs(args.tb_path, exist_ok=True)
    os.makedirs(args.model_path, exist_ok=True)

    # 데이터셋과 데이터로더 설정
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dset = KadisDataset(csv_path, dist_img_dir, transform=transform)
    dataloader_train = DataLoader(dset, batch_size=args.bsize, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 모델 및 손실 함수 설정
    model = CModel(out_dim=args.out_dim).to(device)
    loss_fn = NT_Xent_Sup(args.temperature)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.tm)

    writer = SummaryWriter(args.tb_path)

    start_epoch = 0
    loss_min = float('inf')

    # 체크포인트가 있는 경우 불러오기
    if args.resume:
        ckpt = torch.load(os.path.join(args.model_path, f'{args.model_name}.pt'))
        model.load_state_dict(ckpt['net'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch']
        print(f'=> {start_epoch} 에폭에서 체크포인트 불러오기 완료')

    # 학습 루프
    for epoch in range(start_epoch, args.epoch):
        loss_epoch = train_one_epoch(dataloader_train, model, loss_fn, optimizer, scheduler, device)
        cur_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch [{epoch}/{args.epoch}] Loss: {loss_epoch:.5f} | LR: {cur_lr}')
        writer.add_scalar('Train_loss', loss_epoch, epoch)

        # 모델 저장
        if loss_epoch < loss_min:
            loss_min = loss_epoch
            ckpt = {
                'epoch': epoch,
                'net': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            torch.save(ckpt, os.path.join(args.model_path, f'{args.model_name}.pt'))
            print('=> 체크포인트 저장 완료!')

    writer.close()

def clear_file(args):
    # 기존 파일 삭제
    if os.path.exists(args.tb_path):
        shutil.rmtree(args.tb_path)
        print('=> Tensorboard 파일 삭제 완료')

    print('삭제 완료!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int, default=300, help='에폭 수')
    parser.add_argument('--dset', type=str, default='dist_imgs', help='데이터셋')
    parser.add_argument('--lr', type=float, default=0.0001, help='학습률')
    parser.add_argument('--bsize', type=int, default=32, help='배치 크기')
    parser.add_argument('--psize', type=int, default=224, help='패치 크기')
    parser.add_argument('--pnum', type=int, default=8, help='패치 개수')
    parser.add_argument('--temperature', type=float, default=0.1, help='NT_Xent_Sup 온도')
    parser.add_argument('--out_dim', type=int, default=128, help='모델 출력 차원')
    parser.add_argument('--seed', type=int, default=2022, help='랜덤 시드')
    parser.add_argument('--wd', type=float, default=1e-5, help='가중치 감쇠')
    parser.add_argument('--tm', type=int, default=100, help='Cosine annealing의 최대 반복 횟수')
    parser.add_argument('--sv_path', type=str, default='sav', help='모델 저장 경로')
    parser.add_argument('--tb_path', type=str, default='sav/tb', help='Tensorboard 경로')
    parser.add_argument('--model_path', type=str, default='sav/model', help='모델 저장 경로')
    parser.add_argument('--model_name', type=str, default='iqa-97-msb', help='모델 이름')
    parser.add_argument('--resume', action='store_true', default=False, help='학습 재개')

    args = parser.parse_args()

    clear_file(args)
    main(args)
 """

import torch
import argparse
import os
import random
import torch.optim as optim
import numpy as np
import shutil
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from nt_xent_sup_sparse import NT_Xent_Sup
from dataset import KadisDataset
from model5 import CModel
import torchvision.transforms as transforms
import torch.nn as nn


def initialize_weights(m):
    """모든 레이어에 대해 가중치를 초기화합니다."""
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def adjust_learning_rate(optimizer, epoch, lr_decay_epoch=8):
    """Decay learning rate by a factor every `lr_decay_epoch` epochs."""
    if (epoch + 1) % lr_decay_epoch == 0:
        decay_rate = 0.9 ** (epoch // lr_decay_epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay_rate
    return optimizer


def train_one_epoch(dataloader_train, model, loss_fn, optimizer, scheduler, device):
    """한 에폭 동안의 훈련 루프"""
    loss_epoch = []
    model.train()
    loop = tqdm(dataloader_train, total=len(dataloader_train), desc="Training")

    for batch_idx, (dist, ref, label_dist) in enumerate(loop):
        dist = dist.to(device).float()
        ref = ref.to(device).float()
        label_dist = label_dist.to(device)

        # NaN, Inf 확인
        if torch.isnan(dist).any() or torch.isnan(ref).any() or torch.isnan(label_dist).any():
            print(f"NaN detected in input data at batch {batch_idx}")
            continue

        if torch.isinf(dist).any() or torch.isinf(ref).any() or torch.isinf(label_dist).any():
            print(f"Inf detected in input data at batch {batch_idx}")
            continue

        label_ref = torch.zeros(label_dist.shape[1], device=device)
        label_ref[0] = 1
        label_ref = label_ref.repeat(label_dist.shape[0], 1)

        label = torch.cat([label_dist, label_ref], dim=0)

        z1, z2, _ = model(dist, ref)

        if torch.isnan(z1).any() or torch.isnan(z2).any():
            print(f"NaN detected in z1 or z2 at batch {batch_idx}")
            continue

        loss = loss_fn(z1, z2, label)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"NaN or Inf detected in loss at batch {batch_idx}")
            continue

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)

        optimizer.step()
        scheduler.step()

        loss_epoch.append(loss.item())
        loop.set_postfix(loss=loss.item())

    return np.mean(loss_epoch)


def main(args):
    """훈련 메인 함수"""
    root_dir = 'E:/saTQA/SaTQA/IQA_dataset/kadis700k'
    csv_path = os.path.join(root_dir, 'kadis_new_with_paths.csv')
    dist_img_dir = os.path.join(root_dir, 'E:/saTQA/SaTQA/IQA_dataset/kadis700k/dist_imgs')
    ref_img_dir = os.path.join(root_dir, 'E:/saTQA/SaTQA/IQA_dataset/kadis700k/ref_imgs')

    print(f"CSV 파일 경로: {csv_path}")
    if not os.path.exists(csv_path):
        print("CSV 파일을 찾을 수 없습니다.")
        return

    if args.seed != 0:
        print(f'SEED = {args.seed}')
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.backends.cudnn.deterministic = True

    os.makedirs(args.sv_path, exist_ok=True)
    os.makedirs(args.tb_path, exist_ok=True)
    os.makedirs(args.model_path, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dset = KadisDataset(csv_path, dist_img_dir, transform=transform)
    dataloader_train = DataLoader(dset, batch_size=args.bsize, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CModel(out_dim=args.out_dim).to(device)
    model.apply(initialize_weights)

    loss_fn = NT_Xent_Sup(args.temperature)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.tm)

    writer = SummaryWriter(args.tb_path)

    start_epoch = 0
    loss_min = float('inf')

    if args.resume:
        ckpt = torch.load(os.path.join(args.model_path, f'{args.model_name}.pt'))
        model.load_state_dict(ckpt['net'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch']
        print(f'=> {start_epoch} 에폭에서 체크포인트 불러오기 완료')

    # 학습 루프
    for epoch in range(start_epoch, args.epoch):
        loss_epoch = train_one_epoch(dataloader_train, model, loss_fn, optimizer, scheduler, device)
        cur_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch [{epoch}/{args.epoch}] Loss: {loss_epoch:.5f} | LR: {cur_lr}')
        writer.add_scalar('Train_loss', loss_epoch, epoch)

        # 모델 저장
        if loss_epoch < loss_min:
            loss_min = loss_epoch
            ckpt = {
                'epoch': epoch,
                'net': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            torch.save(ckpt, os.path.join(args.model_path, f'{args.model_name}.pt'))
            print('=> 체크포인트 저장 완료!')

    # 훈련 완료 메시지
    print("훈련이 완료되었습니다.")
    writer.close()


def clear_file(args):
    if os.path.exists(args.tb_path):
        shutil.rmtree(args.tb_path)
        print('=> Tensorboard 파일 삭제 완료')

    print('삭제 완료!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int, default=300, help='에폭 수')
    parser.add_argument('--dset', type=str, default='dist_imgs', help='데이터셋')
    parser.add_argument('--lr', type=float, default=0.0001, help='학습률')
    parser.add_argument('--bsize', type=int, default=32, help='배치 크기')
    parser.add_argument('--psize', type=int, default=224, help='패치 크기')
    parser.add_argument('--pnum', type=int, default=8, help='패치 개수')
    parser.add_argument('--temperature', type=float, default=0.1, help='NT_Xent_Sup 온도')
    parser.add_argument('--out_dim', type=int, default=128, help='모델 출력 차원')
    parser.add_argument('--seed', type=int, default=2022, help='랜덤 시드')
    parser.add_argument('--wd', type=float, default=1e-5, help='가중치 감쇠')
    parser.add_argument('--tm', type=int, default=100, help='Cosine annealing의 최대 반복 횟수')
    parser.add_argument('--sv_path', type=str, default='sav', help='모델 저장 경로')
    parser.add_argument('--tb_path', type=str, default='sav/tb', help='Tensorboard 경로')
    parser.add_argument('--model_path', type=str, default='sav/model', help='모델 저장 경로')
    parser.add_argument('--model_name', type=str, default='iqa-97-msb', help='모델 이름')
    parser.add_argument('--resume', action='store_true', default=False, help='학습 재개')

    args = parser.parse_args()

    clear_file(args)
    main(args)

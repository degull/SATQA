import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch
import pandas as pd

class KadisDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform if transform else transforms.ToTensor()

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # dist_imgs에서 이미지 파일명을 가져옴
        image_file_name = self.data_frame.iloc[idx]['dist_imgs']
        # 절대 경로로 직접 사용
        image_path = image_file_name  # 이미 전체 경로가 포함되어 있다고 가정

        # 경로 확인
        print(f"Looking for image at: {image_path}")
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"{image_path} not found")

        # 이미지 로드
        image = Image.open(image_path)

        # 이미지 변환
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor([1.0])  # 임시로 label_dist를 1.0으로 반환

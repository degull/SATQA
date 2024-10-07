""" import os
import pandas as pd

# 경로 설정
ref_img_dir = r'C:/Users/IIPL02/Desktop/saTQA/SaTQA/IQA_dataset/kadis700k/ref_imgs'
dist_img_dir = r'C:/Users/IIPL02/Desktop/saTQA/SaTQA/IQA_dataset/kadis700k/dist_imgs'

# CSV 파일 경로 설정
csv_path = 'C:/Users/IIPL02/Desktop/saTQA/SaTQA/IQA_dataset/kadis700k/kadis_new_with_paths.csv'

# 원본 이미지 리스트 가져오기
ref_images = os.listdir(ref_img_dir)

# 데이터 저장할 리스트 초기화
data = []

# 이미지에 대해 경로 저장
for ref_img_name in ref_images:
    ref_img_path = os.path.join(ref_img_dir, ref_img_name)
    for i in range(5):  # 5개의 왜곡 이미지가 생성되었으므로
        dist_img_name = f"{ref_img_name.split('.')[0]}_distorted_{i}.png"
        dist_img_path = os.path.join(dist_img_dir, dist_img_name)
        # 원본 이미지와 왜곡 이미지의 경로를 저장
        data.append([ref_img_path, dist_img_path])

# pandas DataFrame으로 변환
df = pd.DataFrame(data, columns=['ref_imgs', 'dist_imgs'])

# CSV 파일로 저장
df.to_csv(csv_path, index=False)

print(f"CSV 파일 생성 완료: {csv_path}")
 """

import pandas as pd

# CSV 파일 불러오기
csv_path = 'E:/saTQA/SaTQA/IQA_dataset/kadis700k/kadis_new_with_paths.csv'
df = pd.read_csv(csv_path)

# 경로 수정
df['dist_imgs'] = df['dist_imgs'].str.replace('C:/Users/IIPL02/Desktop/saTQA', 'E:/saTQA')
df['ref_imgs'] = df['ref_imgs'].str.replace('C:/Users/IIPL02/Desktop/saTQA', 'E:/saTQA')

# 수정된 CSV 파일 저장
df.to_csv(csv_path, index=False)

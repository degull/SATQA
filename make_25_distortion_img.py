import os
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageEnhance
import random
import io
import cv2

# 경로 설정
ref_img_dir = r'C:\Users\IIPL02\Desktop\saTQA\SaTQA\IQA_dataset\kadis700k\ref_imgss'
dist_img_dir = r'C:\Users\IIPL02\Desktop\saTQA\SaTQA\IQA_dataset\kadis700k\dist_imgs'

# dist_imgs 디렉토리 생성
os.makedirs(dist_img_dir, exist_ok=True)

# 왜곡 적용 함수 정의
def apply_gaussian_blur(image):
    return image.filter(ImageFilter.GaussianBlur(radius=random.uniform(1, 5)))

def apply_motion_blur(image):
    return image.filter(ImageFilter.GaussianBlur(radius=random.uniform(1, 5)))  # 간단한 대체 예시

def apply_jpeg_compression(image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG', quality=random.randint(10, 100))  # 랜덤 품질
    return Image.open(img_byte_arr)

def apply_gaussian_noise(image):
    np_image = np.array(image)
    noise = np.random.normal(0, 25, np_image.shape)
    noisy_image = np.clip(np_image + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_image)

def apply_salt_and_pepper_noise(image):
    np_image = np.array(image)
    s_vs_p = 0.5
    amount = 0.04
    out = np.copy(np_image)
    # Salt mode
    num_salt = np.ceil(amount * np_image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in np_image.shape]
    out[coords[0], coords[1], :] = 255

    # Pepper mode
    num_pepper = np.ceil(amount * np_image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in np_image.shape]
    out[coords[0], coords[1], :] = 0
    return Image.fromarray(out)

def apply_color_jitter(image):
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(random.uniform(0.5, 1.5))

def apply_brightness_adjustment(image):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(random.uniform(0.5, 1.5))

def apply_contrast_adjustment(image):
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(random.uniform(0.5, 1.5))

def apply_saturation_adjustment(image):
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(random.uniform(0.5, 1.5))

def apply_sharpness_adjustment(image):
    enhancer = ImageEnhance.Sharpness(image)
    return enhancer.enhance(random.uniform(0.5, 2.0))

def apply_pixelation(image):
    return image.resize((image.size[0] // 10, image.size[1] // 10), Image.NEAREST).resize(image.size, Image.NEAREST)

def apply_random_erasing(image):
    np_image = np.array(image)
    h, w, _ = np_image.shape
    x1 = random.randint(0, w // 2)
    y1 = random.randint(0, h // 2)
    x2 = x1 + random.randint(w // 4, w // 2)
    y2 = y1 + random.randint(h // 4, h // 2)
    np_image[y1:y2, x1:x2] = 255  # 흰색으로 지우기
    return Image.fromarray(np_image)

def apply_color_shift(image):
    np_image = np.array(image)
    shift = np.random.randint(-30, 30, size=3)
    np_image = np.clip(np_image + shift, 0, 255).astype(np.uint8)  # 데이터 타입을 uint8로 변환
    return Image.fromarray(np_image)

def apply_histogram_equalization(image):
    np_image = np.array(image)
    img_yuv = cv2.cvtColor(np_image, cv2.COLOR_RGB2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    return Image.fromarray(cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB))

def apply_gamma_correction(image):
    gamma = random.uniform(0.5, 2.0)
    inv_gamma = 1.0 / gamma
    lookup_table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    np_image = np.array(image)
    return Image.fromarray(lookup_table[np_image])

def apply_image_inversion(image):
    np_image = np.array(image)
    return Image.fromarray(255 - np_image)

def apply_image_flipping(image):
    return image.transpose(Image.FLIP_LEFT_RIGHT)

def apply_cropping(image):
    width, height = image.size
    left = random.randint(0, width // 4)
    top = random.randint(0, height // 4)
    right = random.randint(3 * width // 4, width)
    bottom = random.randint(3 * height // 4, height)
    return image.crop((left, top, right, bottom))

def apply_rotation(image):
    return image.rotate(random.randint(0, 360))

def apply_perspective_distortion(image):
    np_image = np.array(image)
    h, w = np_image.shape[:2]
    src_pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    dst_pts = src_pts + np.random.uniform(-20, 20, src_pts.shape).astype(np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    distorted_image = cv2.warpPerspective(np_image, M, (w, h))
    return Image.fromarray(distorted_image)

def apply_resizing(image):
    new_size = (int(image.size[0] * random.uniform(0.5, 1.5)), 
                 int(image.size[1] * random.uniform(0.5, 1.5)))
    return image.resize(new_size)

def apply_vignette_effect(image):
    np_image = np.array(image)
    h, w = np_image.shape[:2]
    Y, X = np.ogrid[:h, :w]
    center_y, center_x = h / 2, w / 2
    radius = min(center_x, center_y)
    mask = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
    mask = mask / radius
    mask = np.clip(1 - mask, 0, 1)
    for i in range(3):  # RGB 채널에 적용
        np_image[:, :, i] = np.clip(np_image[:, :, i] * mask, 0, 255)
    return Image.fromarray(np_image)

def apply_watermark(image):
    watermark = Image.new('RGBA', image.size, (255, 255, 255, 0))
    watermark_width, watermark_height = watermark.size
    watermark_position = ((image.size[0] - watermark_width) // 2, 
                           (image.size[1] - watermark_height) // 2)
    image.paste(watermark, watermark_position, watermark)
    return image

def apply_mosaic_effect(image):
    np_image = np.array(image)
    h, w, _ = np_image.shape
    block_size = random.randint(8, 32)
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            np_image[y:y + block_size, x:x + block_size] = np_image[y:y + block_size, x:x + block_size].mean(axis=(0, 1))
    return Image.fromarray(np_image)

# 원본 이미지에 왜곡 적용 및 저장
ref_images = os.listdir(ref_img_dir)

for ref_img_name in ref_images:
    img_path = os.path.join(ref_img_dir, ref_img_name)
    image = Image.open(img_path)

    # 각 원본 이미지에 대해 5개의 왜곡 이미지 생성
    distorted_images = [
        apply_gaussian_blur(image),
        apply_motion_blur(image),
        apply_jpeg_compression(image),
        apply_gaussian_noise(image),
        apply_salt_and_pepper_noise(image),
        apply_color_jitter(image),
        apply_brightness_adjustment(image),
        apply_contrast_adjustment(image),
        apply_saturation_adjustment(image),
        apply_sharpness_adjustment(image),
        apply_pixelation(image),
        apply_random_erasing(image),
        apply_color_shift(image),
        apply_histogram_equalization(image),
        apply_gamma_correction(image),
        apply_image_inversion(image),
        apply_image_flipping(image),
        apply_cropping(image),
        apply_rotation(image),
        apply_perspective_distortion(image),
        apply_resizing(image),
        apply_vignette_effect(image),
        apply_watermark(image),
        apply_mosaic_effect(image)
    ]

        # 왜곡 이미지 저장
    for i, distorted_image in enumerate(distorted_images):
        distorted_image.save(os.path.join(dist_img_dir, f"{ref_img_name.split('.')[0]}_distorted_{i}.png"))

    print(f"{ref_img_name}에 대한 모든 왜곡 이미지 생성 완료!")

print("모든 원본 이미지에 대한 왜곡 이미지 생성 완료!")

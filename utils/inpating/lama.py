# -- coding: utf-8 --
""":authors:
    zhuxiaohu
:create_date:
    2026/1/28 3:43
:last_date:
    2026/1/28 3:43
:description:

"""
import os
import time

import cv2
import numpy as np
import torch
from PIL import Image
from simple_lama_inpainting import SimpleLama


def inpaint_image_with_box(image_input, box_coords, output_path, lama=None):
    """
    使用 LaMa 模型修复图片中由坐标框指定的区域。
    * 优化版：仅裁剪相关区域进行推理，大幅提升速度 *

    :param image_input: 输入图片。可以是文件路径(str)，PIL.Image 对象，或 OpenCV 读取的 numpy 数组 (BGR格式)。
    :param box_coords: 一个包含多个 [x, y] 坐标点的列表，定义了需要修复的多边形区域。
                       示例: [[100, 878], [1333, 878], [1333, 993], [100, 993]]
    :param output_path: 修复后图片的保存路径。
    :param lama: (可选) 已经初始化好的 SimpleLama 对象。如果传入，将跳过模型初始化步骤。
    """
    start_time = time.time()

    # 1. 初始化模型 (如果未传入)
    if lama is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            print("警告：正在使用 CPU 运行，速度会非常慢。建议使用 CUDA。")
        print("正在初始化 LaMa 模型 (内部初始化)...")
        lama = SimpleLama(device=device)
        print("模型初始化完成。")
    else:
        # 如果传入了模型，就不需要打印初始化信息了，直接使用
        pass

    # 2. 加载图片并处理不同类型的输入
    if isinstance(image_input, str):
        # 输入是文件路径
        # print(f"从路径加载图片: {image_input}") # 可以注释掉减少日志刷屏
        image = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, np.ndarray):
        # 输入是 NumPy 数组 (通常来自 OpenCV)
        image = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
    elif isinstance(image_input, Image.Image):
        # 输入已经是 PIL Image 对象
        image = image_input.convert("RGB")
    else:
        raise ValueError("不支持的图片输入格式。请输入文件路径、PIL.Image 或 NumPy 数组。")

    # 3. 根据坐标动态生成 Mask (依然在全尺寸上生成，保证坐标准确，随后再裁剪)
    # print("正在根据坐标生成 Mask...")
    # 创建一个与原图同样大小的黑色背景 (np.uint8)
    mask = np.zeros((image.height, image.width), dtype=np.uint8)
    # 将坐标列表转换为 NumPy 数组，格式为 int32
    points = np.array([box_coords], dtype=np.int32)
    # 在黑色背景上填充由坐标定义的多边形区域为白色 (255)
    cv2.fillPoly(mask, pts=points, color=(255))

    # 为了更好地覆盖边缘，可以对 Mask 进行膨胀操作（可选，但推荐）
    kernel = np.ones((15, 15), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)

    # 将 NumPy 格式的 Mask 转换为 PIL Image 格式
    mask_image = Image.fromarray(dilated_mask).convert("L")

    # --- 优化逻辑开始：计算裁剪区域 ---

    # 1. 计算 Mask 的边界框 (x, y, w, h)
    # 这里的 points 是上面定义的坐标点
    x_min_coord = np.min(points[:, :, 0])
    y_min_coord = np.min(points[:, :, 1])
    x_max_coord = np.max(points[:, :, 0])
    y_max_coord = np.max(points[:, :, 1])

    # 2. 定义外扩 Margin (像素)，给予模型上下文信息
    # 如果裁剪太紧，模型无法读取周围背景纹理，修复效果会变差。建议 50px 以上。
    margin = 64

    # 3. 计算安全的裁剪坐标 (防止越界)
    crop_x1 = max(0, int(x_min_coord - margin))
    crop_y1 = max(0, int(y_min_coord - margin))
    crop_x2 = min(image.width, int(x_max_coord + margin))
    crop_y2 = min(image.height, int(y_max_coord + margin))

    # 4. 裁剪原始图片和 Mask
    # PIL crop 格式: (left, upper, right, lower)
    cropped_image = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    cropped_mask = mask_image.crop((crop_x1, crop_y1, crop_x2, crop_y2))

    # 5. 运行修复 (仅针对裁剪区域)
    # print(f"正在处理图片 (Crop size: {cropped_image.size})...")
    inpainted_crop = lama(cropped_image, cropped_mask)

    # 6. 将修复后的区域贴回原图
    # image.paste 会直接修改 image 对象
    image.paste(inpainted_crop, (crop_x1, crop_y1))

    # --- 优化逻辑结束 ---

    # 结果即为粘贴后的 image
    result_image = image

    # 5. 保存结果
    result_image.save(output_path)
    print(f"处理完成，保存至: {output_path} 耗时: {time.time() - start_time:.2f} 秒")


if __name__ == '__main__':

    start_time = time.time()
    image_list = []
    # 替换为你自己的路径
    search_path = r"W:\project\python_project\auto_video\videos\material\7548010999673654547\frame"

    if os.path.exists(search_path):
        for root, dirs, files in os.walk(search_path):
            for file in files:
                if file.endswith(".jpg"):
                    image_list.append(os.path.join(root, file))
    else:
        print(f"路径不存在: {search_path}")

    # --- 关键修改：在循环外部初始化模型 ---
    print("正在预加载 LaMa 模型...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    global_lama = SimpleLama(device=device)
    print("模型预加载完成，开始批量处理...")
    # ----------------------------------

    for img_path in image_list:
        output_path = img_path.replace("frame", "frame_lama_inpainted")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        box = [[100, 878], [1333, 878], [1333, 993], [100, 993]]

        # 将初始化好的 global_lama 传入函数
        inpaint_image_with_box(img_path, box, output_path, lama=global_lama)

    print("全部图片处理完成，总耗时: {:.2f} 秒".format(time.time() - start_time))
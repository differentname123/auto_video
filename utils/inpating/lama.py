# -- coding: utf-8 --
""":authors:
    zhuxiaohu
:create_date:
    2026/1/28 3:43
:last_date:
    2026/1/28 3:43
:description:
    基于 LaMa 的图片与视频去水印/修复工具
"""
import os
import time
import subprocess
import cv2
import numpy as np
import torch
from PIL import Image
from simple_lama_inpainting import SimpleLama

from utils.inpating.lama_utils import convert_normalized_boxes_to_poly


# ==========================================
# 核心辅助函数：仅在内存中处理单帧修复
# ==========================================
def _inpaint_frame_in_memory(image_pil, box_coords_list, lama):
    """
    内存级修复函数，不涉及文件IO。
    :param image_pil: PIL.Image 对象
    :param box_coords_list: 包含多个多边形框的列表 [[[x,y],...], [[x,y],...]]
    :param lama: SimpleLama 模型实例
    :return: 修复后的 PIL.Image 对象
    """
    # 转换为 RGB (防止传入 RGBA 或 BGR)
    image = image_pil.convert("RGB")
    width, height = image.size

    # 遍历当前帧需要修复的所有框（可能同一时间有多个水印）
    for box_coords in box_coords_list:
        # 1. 生成 Mask
        mask = np.zeros((height, width), dtype=np.uint8)
        points = np.array([box_coords], dtype=np.int32)
        cv2.fillPoly(mask, pts=points, color=(255))

        # 膨胀 Mask 以覆盖边缘
        kernel = np.ones((10, 10), np.uint8)  # 视频通常压缩过，膨胀系数可以小一点或保持15
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        mask_image = Image.fromarray(dilated_mask).convert("L")

        # 2. 计算裁剪区域 (Optimization)
        x_min_coord = np.min(points[:, :, 0])
        y_min_coord = np.min(points[:, :, 1])
        x_max_coord = np.max(points[:, :, 0])
        y_max_coord = np.max(points[:, :, 1])

        margin = 50  # 外扩像素
        crop_x1 = max(0, int(x_min_coord - margin))
        crop_y1 = max(0, int(y_min_coord - margin))
        crop_x2 = min(width, int(x_max_coord + margin))
        crop_y2 = min(height, int(y_max_coord + margin))

        # 3. 裁剪
        cropped_image = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        cropped_mask = mask_image.crop((crop_x1, crop_y1, crop_x2, crop_y2))

        # 4. 推理
        try:
            inpainted_crop = lama(cropped_image, cropped_mask)
            # 5. 回贴
            image.paste(inpainted_crop, (crop_x1, crop_y1))
        except Exception as e:
            print(f"LaMa推理出错: {e}，跳过该框")
            continue

    return image

# ==========================================
# 新增功能：视频区间修复函数
# ==========================================
def inpaint_video_intervals(video_path, output_path, repair_info_list, lama=None):
    """
    修复视频指定时间段的指定区域。

    :param video_path: 原视频路径
    :param output_path: 输出视频路径
    :param repair_info_list: 修复信息列表
           [
             {"start": 0, "end": 2000, "boxs": [[[x,y],...], [[x,y],...]]},
             ...
           ]
    :param lama: SimpleLama 模型实例
    """
    if not os.path.exists(video_path):
        print(f"错误：视频文件不存在 {video_path}")
        return

    # 1. 初始化模型
    if lama is None:
        print("初始化 LaMa 模型...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        lama = SimpleLama(device=device)

    # 2. 读取视频信息
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"视频信息: {width}x{height}, FPS: {fps}, 总帧数: {total_frames}")

    # 3. 预处理修复信息：将时间(ms)转换为帧索引范围
    # frames_to_repair 结构: { frame_index: [box_poly1, box_poly2, ...] }
    frames_to_repair = {}

    for info in repair_info_list:
        start_ms = info.get('start', 0)
        end_ms = info.get('end', 0)
        boxes = info.get('boxs', [])  # 这是一个多边形列表

        start_frame = int((start_ms / 1000.0) * fps)
        end_frame = int((end_ms / 1000.0) * fps)

        # 防止越界
        start_frame = max(0, start_frame)
        end_frame = min(total_frames, end_frame)

        for f_idx in range(start_frame, end_frame):
            if f_idx not in frames_to_repair:
                frames_to_repair[f_idx] = []
            # 将该时间段的所有box加入当前帧的待修复列表
            frames_to_repair[f_idx].extend(boxes)

    # 4. 准备临时无声视频输出
    temp_video_path = output_path.replace(".mp4", "_temp_silent.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或者 'avc1'
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

    print("开始逐帧处理视频...")
    start_proc_time = time.time()

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 检查当前帧是否需要修复
        if frame_idx in frames_to_repair:
            # OpenCV (BGR) -> PIL (RGB)
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # 获取当前帧所有需要修复的框
            boxes = frames_to_repair[frame_idx]

            # 调用内存修复函数
            repaired_pil = _inpaint_frame_in_memory(frame_pil, boxes, lama)

            # PIL (RGB) -> OpenCV (BGR)
            frame_out = cv2.cvtColor(np.array(repaired_pil), cv2.COLOR_RGB2BGR)
            out.write(frame_out)
        else:
            # 不需要修复，直接写入原帧
            out.write(frame)

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"进度: {frame_idx}/{total_frames} ({(frame_idx / total_frames) * 100:.1f}%)")

    cap.release()
    out.release()
    print(f"视频画面处理完成，耗时: {time.time() - start_proc_time:.2f}s")

    # 5. 使用 FFmpeg 合并原音频 (System call)
    print("正在合并原始音频...")


    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-i", temp_video_path,
        "-i", video_path,
        "-c:v", "copy",
        "-c:a", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0",
        output_path
    ]

    # 检测系统是否有 ffmpeg
    try:
        # 使用 subprocess.run 运行命令，隐藏过多输出
        result = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            print("FFmpeg 合并失败，可能原视频没有音频或格式不兼容。")
            print("错误信息:", result.stderr.decode('utf-8', errors='ignore'))
            # 如果合并失败，就把无声视频重命名为输出
            if os.path.exists(output_path):
                os.remove(output_path)
            os.rename(temp_video_path, output_path)
        else:
            print("音频合并成功。")
            # 删除临时无声文件
            os.remove(temp_video_path)

    except FileNotFoundError:
        print("错误: 未找到 ffmpeg 命令。请确保 ffmpeg 已安装并添加到环境变量中。")
        print(f"已保留无声视频: {temp_video_path}")
        return

    print(f"全部完成，输出文件: {output_path}")





if __name__ == '__main__':
    # -----------------------
    # 测试用例
    # -----------------------

    # 1. 准备模型 (全局加载一次)
    print("预加载模型...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    global_lama = SimpleLama(device=device)

    # # 2. 定义测试参数
    video_file = r"W:\project\python_project\auto_video\videos\material\7459184511578852646\7459184511578852646_static_cut.mp4"  # 替换为你的视频路径
    output_video = video_file.replace(".mp4", "_inpainted.mp4")

    # 你的输入数据
    input_boxes = [
        {
            "x": 0.1539,
            "y": 0.8795,
            "w": 0.6725,
            "h": 0.0737
        }
    ]

    try:
        # 1. 转换格式
        formatted_boxes = convert_normalized_boxes_to_poly(video_file, input_boxes)

        print("转换后的格式:", formatted_boxes)
        formatted_boxes = [[
            [
                389,
                914
            ],
            [
                1049,
                914
            ],
            [
                1049,
                954
            ],
            [
                389,
                954
            ]
        ]]
        # 2. 结合之前的视频修复函数构建完整参数
        # 假设这些框需要从 0ms 修复到 5000ms
        repair_info = [
            {
                "start": 0,
                "end": 5000,
                "boxs": formatted_boxes  # 直接使用转换后的结果
            }
        ]

        # 3. 调用修复 (前提是你已经定义了 inpaint_video_intervals)
        inpaint_video_intervals(video_file, output_video, repair_info)

    except Exception as e:
        print(f"发生错误: {e}")
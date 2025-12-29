# ocr_engine.py
# -- coding: utf-8 --
import os
import sys
import time
import traceback
from typing import List

import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR
from PIL import Image

from utils.common_utils import time_to_ms


# 1. 移除了 logging 设置，不再使用 logger

class SubtitleOCR:
    _engine_instance = None
    _current_mode = None  # 记录当前是 GPU 还是 CPU

    # 锚定模型根目录：无论在哪里调用，都以本文件所在位置为基准
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODELS_DIR = os.path.join(BASE_DIR, "models_monkt")

    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu

    def _get_engine(self):
        """
        获取引擎实例。如果实例不存在，则初始化。
        """
        # 1. 检查是否已存在且模式匹配
        if SubtitleOCR._engine_instance is not None:
            if SubtitleOCR._current_mode == self.use_gpu:
                return SubtitleOCR._engine_instance
            else:
                # 模式切换了（如 CPU -> GPU），需要销毁重建
                print("Mode changed. Resetting engine...")
                SubtitleOCR._engine_instance = None

        # 2. 路径检查 (使用绝对路径，确保健壮性)
        det_path = os.path.join(self.MODELS_DIR, "detection", "v5", "det.onnx")
        rec_path = os.path.join(self.MODELS_DIR, "languages", "chinese", "rec.onnx")
        keys_path = os.path.join(self.MODELS_DIR, "languages", "chinese", "dict.txt")

        if not all(os.path.exists(p) for p in [det_path, rec_path, keys_path]):
            # 这里记录错误但不抛出崩溃异常，稍后在调用处处理
            print(f"Missing models in {self.MODELS_DIR}")
            return None

        # 3. 加载模型
        try:
            mode_str = "GPU" if self.use_gpu else "CPU"
            print(f"Initializing RapidOCR in {mode_str} mode...")

            engine = RapidOCR(
                det_model_path=det_path,
                cls_model_path=None,
                rec_model_path=rec_path,
                rec_keys_path=keys_path,
                use_angle_cls=False,
                det_use_cuda=self.use_gpu,
                cls_use_cuda=self.use_gpu,
                rec_use_cuda=self.use_gpu
            )
            SubtitleOCR._engine_instance = engine
            SubtitleOCR._current_mode = self.use_gpu
            print("Engine initialized successfully.")
            return engine
        except Exception as e:
            print(f"Failed to initialize engine: {e}")
            traceback.print_exc()
            return None

    def _reset_engine(self):
        """
        强制重置引擎，用于处理底层崩溃后的自愈
        """
        print("Resetting OCR Engine instance due to internal error...")
        SubtitleOCR._engine_instance = None

    def run_batch(self, image_path_list: list, crop_ratio: float = 0.3, confidence: float = 0.8) -> dict:
        """
        执行批量识别。
        保证不抛出异常，返回标准 JSON 结构。
        """
        # 整体结果容器
        response = {
            "code": 0,  # 0 成功, -1 失败
            "message": "success",
            "total_count": len(image_path_list),
            "success_count": 0,
            "failed_count": 0,
            "data": [],  # 存放每张图的结果
            "perf_stats": {}  # 性能统计
        }

        # 用于收集前20个结果用于打印，不放入response
        all_recognized_texts = []

        if not image_path_list:
            response["message"] = "Empty image list"
            return response

        t_start = time.time()

        # 获取引擎 (第一次尝试)
        engine = self._get_engine()
        if engine is None:
            response["code"] = -1
            response["message"] = "Failed to load OCR models. Check model paths."
            return response

        # --- 开始循环处理 (串行处理以保证内存安全) ---
        for img_path in image_path_list:
            item_result = {
                "file_path": img_path,
                "status": "failed",
                "subtitles": [],
                "error_msg": ""
            }

            if not os.path.exists(img_path):
                item_result["error_msg"] = "File not found"
                response["data"].append(item_result)
                response["failed_count"] += 1
                continue

            try:
                # 1. 图像预处理 (Lazy loading, 用完即丢，防止内存积压)
                img = Image.open(img_path)
                width, height = img.size

                # 裁剪
                y_offset = 0
                if 0 < crop_ratio < 1.0:
                    y_offset = int(height * (1 - crop_ratio))
                    crop_area = (0, y_offset, width, height)
                    img_crop = img.crop(crop_area)
                else:
                    img_crop = img

                img_numpy = np.array(img_crop)

                # 2. 推理 (包含重试机制)
                try:
                    ocr_result, _ = engine(img_numpy)
                except Exception as e_engine:
                    # !!! 关键点：如果推理报错，可能是引擎坏了，尝试重置一次 !!!
                    print(f"Inference error on {os.path.basename(img_path)}, retrying...")
                    self._reset_engine()
                    engine = self._get_engine()  # 重新获取
                    if engine:
                        ocr_result, _ = engine(img_numpy)  # 再次尝试
                    else:
                        raise Exception("Engine reload failed")

                # 3. 格式化结果
                if ocr_result:
                    valid_subs = []
                    for line in ocr_result:
                        if line and len(line) == 3 and line[2] > confidence:
                            box, text, score = line[0], line[1], line[2]

                            # 坐标还原 + 类型转换 (防止 JSON 报错)
                            final_box = [[int(p[0]), int(p[1] + y_offset)] for p in box]

                            valid_subs.append({
                                "text": text,
                                "box": final_box,
                                "score": float(score)
                            })

                            # 收集识别到的文本，用于最后的打印 summary
                            all_recognized_texts.append(text)

                    item_result["subtitles"] = valid_subs

                item_result["status"] = "success"
                response["success_count"] += 1

            except Exception as e:
                # 捕获单张图片的任何异常（如图片损坏、分辨率过大导致的问题）
                item_result["error_msg"] = str(e)
                item_result["status"] = "error"
                response["failed_count"] += 1
                # 打印堆栈以便调试，但不中断程序
                # traceback.print_exc()

            response["data"].append(item_result)

        # --- 统计结束 ---
        t_end = time.time()
        total_time = t_end - t_start

        response["perf_stats"] = {
            "total_time_sec": float(f"{total_time:.4f}"),
            "avg_time_per_img_sec": float(f"{total_time / len(image_path_list):.4f}") if image_path_list else 0
        }

        # 2. 不使用 logger，直接 print 关键信息
        print("=" * 40)
        print(f"处理图片的数量: {response['total_count']}")
        print(f"耗时: {total_time:.4f} 秒")
        print(f"成功数量: {response['success_count']}")
        print(f"前20个字符串的识别结果: {all_recognized_texts[:20]}")
        print("=" * 40)

        return response

def analyze_and_filter_boxes(
        boxes: List[List[List[int]]],
        height_tolerance_ratio: float = 0.3,
        y_pos_tolerance_ratio: float = 0.25,
        z_score_threshold: float = 2.0
) -> List[List[List[int]]]:
    """
    【三阶段优化版】分析并过滤字幕框，增加下底位置筛选。

    Args:
        boxes (List[List[List[int]]]): 所有检测到的字幕框。
        height_tolerance_ratio (float): 第一阶段过滤中，框高与中位数的最大允许偏差比例。
        y_pos_tolerance_ratio (float): 第二阶段过滤中，框下底与中位下底的最大允许偏差比例
                                       (以中位数高度为基准)。
        z_score_threshold (float): 第三阶段过滤中，用于判断是否为异常值的Z分数阈值。

    Returns:
        List[List[List[int]]]: 过滤后的高质量字幕框列表。
    """
    if len(boxes) < 5:  # 数据太少，统计无意义，直接返回
        print("[过滤] 检测到的框数量过少，跳过复杂过滤。")
        return boxes

    # --- 阶段 1: 基于高度中位数的粗筛 ---
    heights = []
    for box in boxes:
        y_coords = [p[1] for p in box]
        heights.append(max(y_coords) - min(y_coords))

    median_height = np.median(heights)
    if median_height == 0: return []  # 避免后续除零错误

    height_diff_threshold = median_height * height_tolerance_ratio

    height_filtered_boxes = []
    print(f"[过滤-阶段1: 高度] 以中位高度 {median_height:.2f} 为基准 (容忍度: {height_diff_threshold:.2f}px)。")
    for i, box in enumerate(boxes):
        if abs(heights[i] - median_height) <= height_diff_threshold:
            height_filtered_boxes.append(box)
        else:
            print(f"  - 剔除高度异常框: 高度为 {heights[i]}, 与中位数差异过大。")

    print(f"[过滤-阶段1: 高度] 完成，剩余 {len(height_filtered_boxes)} 个框进入下一阶段。")
    if len(height_filtered_boxes) < 3:
        return height_filtered_boxes

    # --- 阶段 2: 基于下底位置的中筛 (新增筛选逻辑) ---
    bottom_ys = [max(p[1] for p in box) for box in height_filtered_boxes]
    median_bottom_y = np.median(bottom_ys)

    # 位置容忍度以中位高度为基准，更具自适应性
    y_pos_diff_threshold = median_height * y_pos_tolerance_ratio

    position_filtered_boxes = []
    print(f"[过滤-阶段2: 位置] 以中位下底 {median_bottom_y:.2f} 为基准 (容忍度: {y_pos_diff_threshold:.2f}px)。")
    for i, box in enumerate(height_filtered_boxes):
        if abs(bottom_ys[i] - median_bottom_y) <= y_pos_diff_threshold:
            position_filtered_boxes.append(box)
        else:
            print(f"  - 剔除位置异常框: 下底为 {bottom_ys[i]} (期望: {median_bottom_y:.2f})，差异过大。")

    print(f"[过滤-阶段2: 位置] 完成，剩余 {len(position_filtered_boxes)} 个框进入精筛。")
    if len(position_filtered_boxes) < 3:
        return position_filtered_boxes

    # --- 阶段 3: 对精筛后的框进行Z-score统计分析 ---
    properties = []
    for box in position_filtered_boxes:
        y_coords = [p[1] for p in box]
        min_y, max_y = min(y_coords), max(y_coords)
        properties.append({'height': max_y - min_y, 'center_y': min_y + (max_y - min_y) / 2})

    clean_heights = np.array([p['height'] for p in properties])
    clean_center_ys = np.array([p['center_y'] for p in properties])

    mean_height, std_height = np.mean(clean_heights), np.std(clean_heights)
    mean_center_y, std_center_y = np.mean(clean_center_ys), np.std(clean_center_ys)

    final_good_boxes = []
    print(f"[过滤-阶段3: 精筛] 以稳定数据为基准进行Z-score分析。")
    for i, box in enumerate(position_filtered_boxes):
        prop = properties[i]
        # 计算 Z-score, 避免标准差为0时除零
        height_z = abs(prop['height'] - mean_height) / std_height if std_height > 0 else 0
        center_y_z = abs(prop['center_y'] - mean_center_y) / std_center_y if std_center_y > 0 else 0

        if height_z < z_score_threshold and center_y_z < z_score_threshold:
            final_good_boxes.append(box)
        else:
            print(f"  - 剔除Z-score异常框 (高度Z: {height_z:.2f}, 中心Y Z: {center_y_z:.2f})")

    print(f"[过滤] 所有流程完成，最终保留 {len(final_good_boxes)} 个高质量字幕框。")
    return final_good_boxes


def draw_box_on_images(image_paths: List[str], box_coords: List[List[int]]):
    """
    将一个指定的包围框绘制到一组图片上并覆盖保存。
    """
    # 从[[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]中获取左上角和右下角点
    pt1 = tuple(box_coords[0])  # (min_x, min_y)
    pt2 = tuple(box_coords[2])  # (max_x, max_y)
    color = (0, 255, 0)  # 绿色 (BGR)
    thickness = 2

    for path in image_paths:
        try:
            img = cv2.imread(path)
            if img is None:
                continue
            cv2.rectangle(img, pt1, pt2, color, thickness)
            cv2.imwrite(path, img)  # 覆盖保存
        except Exception as e:
            print(f"绘制 '{os.path.basename(path)}' 时出错: {e}")


def adjust_subtitle_box(video_path, final_box):
    """
    将字幕框左右边距至少设为视频宽度的 10%，
    但如果原框更宽就不再缩窄它。

    参数:
        video_path: 视频文件路径
        final_box: 原始字幕框，格式 [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]

    返回:
        (top_left, bottom_right)：调整后的左上角和右下角坐标
    """
    # 1. 打开视频，获取分辨率
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频文件: {video_path}")
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # 2. 阈值：左右各保留 10%
    thresh_left  = int(width * 0.1)
    thresh_right = int(width * 0.9)

    # 3. 原始框的最小/最大 x，和最小/最大 y
    xs = [pt[0] for pt in final_box]
    ys = [pt[1] for pt in final_box]
    orig_x_min = min(xs)
    orig_x_max = max(xs)
    y_top      = min(ys)
    y_bottom   = max(ys)

    # 4. 如果原框比 阈值 宽度小则扩张，否则保留
    new_x_left  = min(orig_x_min, thresh_left)
    new_x_right = max(orig_x_max, thresh_right)

    # 5. 构造返回值
    top_left     = [new_x_left,  y_top]
    bottom_right = [new_x_right, y_bottom]

    return top_left, bottom_right, width, height



def find_overall_subtitle_box_target_number(
    video_path: str,
    merged_timerange_list: list[dict],
    num_samples: int = 10,
    output_dir = 'temp_dir'
):
    """
    主函数，找到包围视频字幕的最小框，并将框绘制到所有抽帧图片上。
    且保证抽取到 num_samples 帧（或所有符合区间的帧）。
    如果字幕框高度超过视频高度的10%，则返回 None。
    :param video_path: 视频文件路径
    :param merged_timerange_list: [{ "startTime": "00:00:00.205", "endTime": "00:00:09.060" }, ...]
    :param num_samples: 希望抽取的帧数
    :return: 最终包围框顶点列表 [[x1, y1], ..., [x4, y4]] 或 None
    """
    os.makedirs(output_dir, exist_ok=True)
    ocr = SubtitleOCR(use_gpu=True)
    # --- 检查视频文件 ---
    if not os.path.exists(video_path):
        print(f"错误: 视频文件未找到 '{video_path}'")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 '{video_path}'")
        return None

    # 获取视频基本信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if total_frames <= 0 or fps <= 0 or video_height <= 0:
        print("错误: 无法获取视频信息，抽帧失败。")
        cap.release()
        return None

    # --- 预处理：将所有时间区间转为毫秒，并对应到帧索引 ---
    valid_frames = set()
    for tr in merged_timerange_list:
        start_ms = time_to_ms(tr['startTime'])
        end_ms = time_to_ms(tr['endTime'])
        if end_ms <= start_ms:
            continue
        # 对应的帧索引区间
        start_idx = int(np.ceil(start_ms / 1000 * fps))
        end_idx   = int(np.floor(end_ms   / 1000 * fps))
        # 限制在 [0, total_frames-1]
        start_idx = max(0, start_idx)
        end_idx   = min(total_frames - 1, end_idx)
        valid_frames.update(range(start_idx, end_idx + 1))

    valid_frames = sorted(valid_frames)
    if not valid_frames:
        print("错误: 没有任何帧落在指定的时间区间内。")
        cap.release()
        return None

    # --- 从 valid_frames 中均匀抽取 num_samples 帧 ---
    num_to_pick = min(num_samples, len(valid_frames))
    step = len(valid_frames) / num_to_pick
    sample_indices = [valid_frames[int(i * step)] for i in range(num_to_pick)]

    # --- 阶段 1: 按 sample_indices 抽帧并保存 ---
    print(f"\n[阶段 1] 从 {len(valid_frames)} 个符合区间的帧中抽取 {len(sample_indices)} 帧...")
    saved_frame_paths = []
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            print(f"警告: 无法读取第 {idx} 帧。")
            continue
        path = os.path.join(output_dir, f"frame_{idx}.jpg")
        cv2.imwrite(path, frame)
        saved_frame_paths.append(path)
    cap.release()
    # print(f"[阶段 1] 抽帧完成。共保存 {len(saved_frame_paths)} 帧图片。")

    if not saved_frame_paths:
        print("未能提取任何帧。")
        return None


    # --- 阶段 2: 对抽出的帧进行字幕检测 ---
    # print(f"\n[阶段 2] 开始检测 {len(saved_frame_paths)} 张图片的字幕框...")
    result_json = ocr.run_batch(saved_frame_paths)

    detected_boxes = [sub.get("box", []) for item in result_json.get("data", []) for sub in item.get("subtitles", [])]

    print(f"[阶段 2] 检测完成。共检测到 {len(detected_boxes)} 个字幕框。")

    if not detected_boxes:
        print("未找到任何字幕框。")
        return None


    # --- 阶段 3: 分析并计算最终包围框 ---
    print("\n[阶段 3] 开始分析字幕框并计算最终包围区域...")
    good_boxes = analyze_and_filter_boxes(detected_boxes)

    if not good_boxes:
        print("\n[结果] 所有检测到的框都被过滤为异常值。图片已保存在 'temp_dir' 目录中，但未做任何修改。")
        return

    print(f"[阶段 3] 过滤后剩余 {len(good_boxes)} 个有效字幕框。")

    all_points = np.array([point for box in good_boxes for point in box])
    final_box = [
        [int(np.min(all_points[:, 0])), int(np.min(all_points[:, 1]))],  # min_x, min_y
        [int(np.max(all_points[:, 0])), int(np.min(all_points[:, 1]))],  # max_x, min_y
        [int(np.max(all_points[:, 0])), int(np.max(all_points[:, 1]))],  # max_x, max_y
        [int(np.min(all_points[:, 0])), int(np.max(all_points[:, 1]))]  # min_x, max_y
    ]
    print(f"[阶段 3] 计算出的最终包围框: {final_box}")
    return final_box

    # --- 阶段 4: 将最终包围框绘制到所有抽取的帧上 ---
    print(f"\n[阶段 4] 正在将最终包围框绘制到 '{output_dir}' 目录下的 {len(saved_frame_paths)} 张图片上...")
    draw_box_on_images(saved_frame_paths, final_box)
    print("[阶段 4] 绘制完成。")

    # --- 任务结束 ---
    print("\n" + "=" * 60)
    print("任务成功！")
    print(f"最终的字幕包围框为: {final_box}")
    print(f"带有包围框的图片已全部保存在 '{output_dir}' 目录中。")
    print("=" * 60)


# ================= 测试调用 =================
if __name__ == "__main__":
    # 模拟路径 (请确保这些文件存在，或修改为你的真实路径进行测试)
    # 故意包含一个不存在的图片来测试健壮性
    test_images = [
        r"W:\project\python_project\watermark_remove\common_utils\video_scene\scenes\52.267\frame_1565_time_00_00_52_166.png",
        r"W:\project\python_project\watermark_remove\common_utils\video_scene\scenes\52.267\frame_1565_time_00_00_52_166.png",
        r"W:\project\python_project\watermark_remove\common_utils\video_scene\scenes\52.267\frame_1565_time_00_00_52_166.png",
        r"W:\project\python_project\watermark_remove\common_utils\video_scene\scenes\52.267\frame_1565_time_00_00_52_166.png",

    ]

    print("\n--- Starting Safe Batch OCR ---")
    # # 这一步绝对不会报错，只会返回 JSON
    for i in range(5):
        ocr = SubtitleOCR(use_gpu=True)

        print(f"\n--- OCR Attempt {i + 1} ---")
        result_json = ocr.run_batch(test_images)
        print(result_json)
        # 获取所有的box，保存到box_list
        box_list = [sub.get("box", []) for item in result_json.get("data", []) for sub in item.get("subtitles", [])]




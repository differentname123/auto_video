# ocr_engine.py
# -- coding: utf-8 --
import os
import time
import gc
import traceback
import cv2
import numpy as np
from PIL import Image
from rapidocr_onnxruntime import RapidOCR

from utils.common_utils import safe_process_limit


def _get_model_paths():
    """获取模型路径，基于当前文件位置"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models_monkt")

    det_path = os.path.join(models_dir, "detection", "v5", "det.onnx")
    rec_path = os.path.join(models_dir, "languages", "chinese", "rec.onnx")
    keys_path = os.path.join(models_dir, "languages", "chinese", "dict.txt")

    return det_path, rec_path, keys_path


def _init_engine(use_gpu: bool):
    """初始化引擎并返回实例"""
    det_path, rec_path, keys_path = _get_model_paths()

    if not all(os.path.exists(p) for p in [det_path, rec_path, keys_path]):
        print(f"Missing models in {os.path.dirname(det_path)}...")
        return None

    try:
        engine = RapidOCR(
            det_model_path=det_path,
            cls_model_path=None,
            rec_model_path=rec_path,
            rec_keys_path=keys_path,
            use_angle_cls=False,
            det_use_cuda=use_gpu,
            cls_use_cuda=use_gpu,
            rec_use_cuda=use_gpu
        )
        print(f"Engine initialized successfully (GPU={use_gpu}).")
        return engine
    except Exception as e:
        print(f"Failed to initialize engine: {e}")
        traceback.print_exc()
        return None


def _select_best_subtitle_strict(
        raw_ocr_lines: list,
        img_h: int,
        img_w: int,
        bottom_ratio_in_crop: float = 0.0,
        rect_ang_thresh: float = 10.0,
        rect_ratio_thresh: float = 0.8,
        aspect_ratio_thresh: float = 2.0,
        width_ratio_thresh: float = 0.1
):
    """
    严格筛选逻辑（纯函数，无状态）
    """
    if not raw_ocr_lines:
        return None

    min_y_limit = img_h * bottom_ratio_in_crop
    candidates = []

    # 1. 底部筛选
    for line in raw_ocr_lines:
        box = np.array(line[0], dtype=np.float32)
        if np.min(box[:, 1]) >= min_y_limit:
            candidates.append({
                "box": box,
                "text": line[1],
                "score": line[2]
            })

    if not candidates:
        return None

    # 2. 形状 & 几何筛选
    filtered = []
    min_width = img_w * width_ratio_thresh
    center_x = img_w / 2.0

    for item in candidates:
        box = item["box"]
        if box.shape[0] != 4:
            continue

        xs = box[:, 0]
        width = xs.max() - xs.min()

        if width < min_width: continue  # 宽度不够
        if not (xs.min() < center_x < xs.max()): continue  # 未跨越中心线

        rect = cv2.minAreaRect(box)
        angle = abs(rect[2])
        effective_angle = min(angle, 90.0 - angle)

        if effective_angle > rect_ang_thresh: continue  # 角度太大

        area_poly = cv2.contourArea(box)
        box_w, box_h = rect[1]
        area_rect = box_w * box_h

        if area_rect <= 0: continue
        if (area_poly / area_rect) < rect_ratio_thresh: continue  # 矩形度不够

        if aspect_ratio_thresh is not None:
            ar = max(box_w, box_h) / (min(box_w, box_h) + 1e-6)
            if ar < aspect_ratio_thresh: continue  # 宽高比不够

        filtered.append(item)

    if not filtered:
        return None

    if len(filtered) == 1:
        return filtered[0]

    # 3. 打分排序
    Y_WEIGHT = 1.0
    X_WEIGHT = 10.0

    def calculate_score(item):
        box = item["box"]
        cy = np.mean(box[:, 1])
        y_score = (cy / img_h) * Y_WEIGHT
        cx = np.mean(box[:, 0])
        x_pen = (abs(cx - center_x) / img_w) * X_WEIGHT
        return y_score - x_pen

    best = max(filtered, key=calculate_score)
    return best

@safe_process_limit(limit=4, name="run_subtitle_ocr")
def run_subtitle_ocr(image_path_list: list, use_gpu: bool = True, crop_ratio: float = 0.5,
                     confidence: float = 0.8) -> dict:
    """
    对外的主函数：无状态、每次运行初始化、运行完清理。
    """
    all_start_time = time.time()
    print(f"Starting batch OCR for {len(image_path_list)} images (Stateless Mode)...")

    response = {
        "code": 0,
        "message": "success",
        "total_count": len(image_path_list),
        "success_count": 0,
        "failed_count": 0,
        "data": [],
        "perf_stats": {}
    }

    all_recognized_texts = []

    if not image_path_list:
        response["message"] = "Empty image list"
        return response

    # --- 1. 初始化模型 (每次调用都重新初始化) ---
    engine = _init_engine(use_gpu)
    if engine is None:
        response["code"] = -1
        response["message"] = "Failed to load OCR models."
        return response

    t_start = time.time()

    # --- 2. 批量处理 ---
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
            # 图片预处理
            img = Image.open(img_path)
            width, height = img.size

            y_offset = 0
            if 0 < crop_ratio < 1.0:
                y_offset = int(height * (1 - crop_ratio))
                crop_area = (0, y_offset, width, height)
                img_crop = img.crop(crop_area)
            else:
                img_crop = img

            img_numpy = np.array(img_crop)
            crop_h, crop_w = img_numpy.shape[:2]

            # 推理
            try:
                result_raw, _ = engine(img_numpy)
                ocr_result = result_raw if result_raw else []
            except Exception as e_inf:
                # 如果推理过程中崩了，尝试本地简单重建一次（可选，防止单张图卡死整个批次）
                print(f"Inference error on {os.path.basename(img_path)}: {e_inf}")
                item_result["error_msg"] = f"Inference Error: {e_inf}"
                ocr_result = []

            # 结果筛选
            if ocr_result:
                # 初步置信度过滤
                high_conf_lines = [
                    line for line in ocr_result
                    if line and len(line) == 3 and line[2] > confidence
                ]

                # 几何筛选
                best_sub = _select_best_subtitle_strict(
                    high_conf_lines,
                    crop_h,
                    crop_w,
                    bottom_ratio_in_crop=0.0,
                    rect_ang_thresh=10.0,
                    rect_ratio_thresh=0.8,
                    aspect_ratio_thresh=1,
                    width_ratio_thresh=0.1
                )

                if best_sub:
                    text = best_sub["text"]
                    box = best_sub["box"]
                    score = best_sub["score"]

                    # 坐标还原
                    final_box = [[int(p[0]), int(p[1] + y_offset)] for p in box]

                    item_result["subtitles"].append({
                        "text": text,
                        "box": final_box,
                        "score": float(score)
                    })
                    all_recognized_texts.append(text)

            item_result["status"] = "success"
            response["success_count"] += 1

        except Exception as e:
            item_result["error_msg"] = str(e)
            item_result["status"] = "error"
            response["failed_count"] += 1
            # print(f"Error processing {img_path}: {e}")

        response["data"].append(item_result)

    # --- 3. 资源清理 ---
    # 显式删除引擎并强制GC，确保内存释放
    try:
        del engine
    except:
        pass
    gc.collect()
    if use_gpu:
        # 如果是GPU模式，有时候需要清空CUDA缓存(依赖torch，这里假设rapidocr后端处理了，或者只靠gc)
        pass

        # --- 4. 统计与返回 ---
    t_end = time.time()
    total_time = t_end - t_start
    response["perf_stats"] = {
        "total_time_sec": float(f"{total_time:.4f}"),
        "avg_time_per_img_sec": float(f"{total_time / len(image_path_list):.4f}") if image_path_list else 0
    }

    print("=" * 40)
    print(f"处理图片的数量: {response['total_count']}")
    print(f"运行耗时: {total_time:.4f} 秒 总共耗时{time.time() - all_start_time:.4f}秒")
    print(f"成功数量: {response['success_count']}")
    print(f"前20个字符串的识别结果: {all_recognized_texts[:20]}")
    print("=" * 40)

    return response


if __name__ == "__main__":
    # 模拟路径 (请确保这些文件存在，或修改为你的真实路径进行测试)
    # 故意包含一个不存在的图片来测试健壮性
    test_images = [
        r"C:\Users\zxh\Desktop\temp\申诉截图3.png"

    ]
    # 扫描W:\project\python_project\auto_video\videos\material\7590376286820814107\frame下面所有的png文件
    image_list = []
    for root, dirs, files in os.walk(r"W:\project\python_project\auto_video\videos\material\7590376286820814107\frame"):
        for file in files:
            if file.endswith(".jpg"):
                image_list.append(os.path.join(root, file))




    print("\n--- Starting Safe Batch OCR ---")
    # # 这一步绝对不会报错，只会返回 JSON
    for i in range(5):
        print(f"\n--- OCR Attempt {i + 1} ---")
        result_json = run_subtitle_ocr(test_images, crop_ratio=0.5)
        print(result_json)
        # 获取所有的box，保存到box_list
        box_list = [sub.get("box", []) for item in result_json.get("data", []) for sub in item.get("subtitles", [])]
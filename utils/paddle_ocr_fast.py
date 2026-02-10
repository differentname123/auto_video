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
    det_path = os.path.join(models_dir, "detection", "v5", "ch_PP-OCRv5_mobile_det.onnx")

    # det_path = os.path.join(models_dir, "detection", "v5", "det_v4.onnx")
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
            rec_model_path=rec_path,
            rec_keys_path=keys_path,
            cls_model_path=None,  # 显式禁用方向分类器（绝对不要加载，字幕通常都是正的）
            use_angle_cls=False,  # 再次确认关闭角度分类

            # --- 核心提速配置 (检测部分) ---
            # 限制检测时图片的长边长度。
            # 默认是 960。对于简单的大字字幕，降低到 320 或 480 即可。
            # 这会极大减少计算量，速度提升 2~3 倍。
            det_limit_side_len=480,

            # 检测框阈值，保持默认或适当降低以防止漏检
            det_db_box_thresh=0.5,
            det_db_unclip_ratio=1.6,

            # 硬件加速配置
            det_use_cuda=use_gpu,
            cls_use_cuda=use_gpu,
            rec_use_cuda=use_gpu,

            # 预热选项 (可选，稍微增加启动时间但稳定后续速度)
            det_use_dml=False,  # 如果是Windows非N卡可以用True，N卡用False
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


def run_fast_det_rec_ocr(image_path_list: list, use_gpu: bool = True, engine=None, max_width: int = 1500,
                         score_threshold: float = 0.5) -> dict:
    """
    极速检测+识别模式：针对条状字幕图优化。
    通过限制输入图片尺寸（max_width）来加速检测器运行，并合并结果。

    :param max_width: 限制图片的最大宽度，超过此宽度将按比例缩放。推荐值 1000~2000。
    """
    t_start = time.time()

    # 1. 引擎初始化逻辑
    local_engine = False
    if engine is None:
        # 注意：此处假设外部有 _init_engine 函数，保持原逻辑不变
        try:
            engine = _init_engine(use_gpu)
            local_engine = True
        except NameError:
            # 防止上下文中没有 _init_engine 的报错处理（仅作代码完整性示意）
            pass

    if engine is None:
        return {"code": -1, "message": "Model Init Failed"}

    results = []
    success_count = 0

    # print(f"Starting Fast Det+Rec OCR for {len(image_path_list)} images (Max Width: {max_width}px)...")

    for img_path in image_path_list:
        item_result = {"file": img_path, "status": "failed"}

        if not os.path.exists(img_path):
            item_result["status"] = "not_found"
            results.append(item_result)
            continue

        try:
            # 1. 读取图片
            with open(img_path, 'rb') as f:
                img_bytes = np.frombuffer(f.read(), np.uint8)

            img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
            if img is None:
                item_result["status"] = "read_error"
                results.append(item_result)
                continue

            h, w = img.shape[:2]

            # 2. 核心加速点：图片预缩放
            scale_ratio = 1.0  # 初始化缩放比例，用于后续坐标还原
            if w > max_width:
                # 保持宽高比进行缩放
                scale_ratio = max_width / w
                new_h = int(h * scale_ratio)
                img_resized = cv2.resize(img, (max_width, new_h), interpolation=cv2.INTER_LINEAR)
            else:
                img_resized = img

            # 3. 推理 (开启检测，关闭分类)
            # RapidOCR 会自动处理图像数据格式
            ocr_result, _ = engine(img_resized, use_det=True, use_cls=False, use_rec=True)

            if ocr_result:
                # --- 结果清洗与合并 ---
                ocr_result.sort(key=lambda x: x[0][0][0])

                combined_text = []
                min_score = 1.0

                # 用于收集所有有效框的坐标点
                all_x_coords = []
                all_y_coords = []

                for item in ocr_result:
                    box, text, score = item

                    if score < score_threshold:
                        continue

                    combined_text.append(text)
                    min_score = min(min_score, score)

                    # 收集该文本块的四个顶点坐标
                    for point in box:
                        all_x_coords.append(point[0])
                        all_y_coords.append(point[1])

                final_text = "".join(combined_text)

                # 只有当识别到文本且有坐标时才计算包围框
                if final_text and all_x_coords:
                    # 1. 找到所有点的极值（最小外接矩形）
                    min_x = min(all_x_coords)
                    max_x = max(all_x_coords)
                    min_y = min(all_y_coords)
                    max_y = max(all_y_coords)

                    # 2. 将坐标还原回原图尺寸 (除以缩放比例)
                    # 构造四个顶点：[左上, 右上, 右下, 左下]
                    merged_box = [
                        [min_x / scale_ratio, min_y / scale_ratio],
                        [max_x / scale_ratio, min_y / scale_ratio],
                        [max_x / scale_ratio, max_y / scale_ratio],
                        [min_x / scale_ratio, max_y / scale_ratio]
                    ]

                    item_result.update({
                        "text": final_text,
                        "score": float(min_score),
                        "box": merged_box,  # 新增：合并后的最小包围框
                        "status": "success"
                    })
                    success_count += 1
                else:
                    item_result["status"] = "filtered"
            else:
                item_result["status"] = "empty"

        except Exception as e:
            item_result["status"] = "error"
            item_result["msg"] = str(e)

        results.append(item_result)

    # 4. 清理资源
    if local_engine:
        del engine
        gc.collect()

    t_end = time.time()
    total_time = t_end - t_start
    return {
        "code": 0,
        "total": len(image_path_list),
        "success": success_count,
        "time_cost": f"{total_time:.4f}s",
        "avg_time": f"{total_time / len(image_path_list):.4f}s" if image_path_list else 0,
        "data": results
    }

if __name__ == "__main__":
    image_list = []
    for root, dirs, files in os.walk(r"W:\project\python_project\auto_video\videos\material\7597599415717615476\test_subtitle_box"):
        for file in files:
            if file.endswith(".jpg"):
                if "cropped" in file:
                    continue
                image_list.append(os.path.join(root, file))


    print("\n--- 极速识别模式 (纯识别) ---")
    global_engine = _init_engine(use_gpu=True)
    total_cost = 0.0
    for i in range(1):
        result = run_fast_det_rec_ocr(image_list, engine=global_engine)
        print(f"总耗时: {result['time_cost']}")
        total_cost += float(result['time_cost'][:-1])
        for item in result['data']:
            if item.get('text'):
                print(f"文件名: {os.path.basename(item['file'])}")
                print(f"识别文字: 【{item['text']}】 (Score: {item['score']:.2f})")
                print("-" * 20)
        print(result['data'])

    print(f"5次平均耗时: {total_cost / 5:.4f}s")
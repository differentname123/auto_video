# ocr_engine.py
# -- coding: utf-8 --
import os
import sys
import time
import traceback
import numpy as np
from rapidocr_onnxruntime import RapidOCR
from PIL import Image


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
    ocr = SubtitleOCR(use_gpu=True)
    result_json = ocr.run_batch(test_images)
    # # 这一步绝对不会报错，只会返回 JSON
    for i in range(5):
        print(f"\n--- OCR Attempt {i + 1} ---")
        result_json = ocr.run_batch(test_images)


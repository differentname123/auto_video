import cv2
import numpy as np
import os
import uuid
import traceback

# 假设这两个函数在 utils.paddle_ocr_debug 中
from utils.paddle_ocr_fast import run_fast_det_rec_ocr, _init_engine


def crop_polygon(img, points):
    """裁剪图像，利用切片特性简化边界处理"""
    x, y, w, h = cv2.boundingRect(np.array(points, dtype=np.int32))
    # 确保左上角不小于0即可，右下角越界Python切片会自动截断
    x, y = max(0, x), max(0, y)
    return img[y:y + h, x:x + w]


def is_similar_image(img1, img2, threshold=30):
    """判断两张图片是否相似 (尺寸缩放 + 灰度差值)"""
    if img1 is None or img2 is None or img1.shape != img2.shape:
        return False

    # 转灰度
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 缩放加速比对 (保持比例)
    h, w = g1.shape
    new_w = 64
    new_h = int(new_w * h / w) if w > 0 else 64

    s1 = cv2.resize(g1, (new_w, new_h))
    s2 = cv2.resize(g2, (new_w, new_h))

    return np.mean(cv2.absdiff(s1, s2)) < threshold


def video_ocr_processor(video_path, ocr_info, similarity_threshold=25):
    global_engine = _init_engine(use_gpu=True)
    video_dir = os.path.dirname(os.path.abspath(video_path))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_results = []

    try:
        for info in ocr_info:
            boxes = info['boxs']
            start_frame = int((info['start'] / 1000) * fps)
            end_frame = int((info['end'] / 1000) * fps)

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            # 状态缓存: [last_image, last_text]
            # 使用列表索引直接对应 box 索引
            cache = [[None, ""] for _ in boxes]

            for curr_frame_idx in range(start_frame, end_frame + 1):
                ret, frame = cap.read()
                if not ret: break

                frame_res = {
                    "frame_index": curr_frame_idx,
                    "timestamp": cap.get(cv2.CAP_PROP_POS_MSEC),
                    "ocr_data": {}
                }

                # 待处理任务: 存储 (box_index, cropped_img, temp_file_path)
                pending_tasks = []

                # 1. 筛选需要 OCR 的区域
                for i, points in enumerate(boxes):
                    crop = crop_polygon(frame, points)

                    if is_similar_image(crop, cache[i][0], threshold=similarity_threshold):
                        # 相似则复用
                        frame_res["ocr_data"][i] = cache[i][1]
                    else:
                        # 不相似则准备 OCR，同时更新缓存图像
                        temp_name = f"tmp_{uuid.uuid4().hex}.jpg"
                        temp_path = os.path.join(video_dir, temp_name)
                        pending_tasks.append((i, crop, temp_path))
                        cache[i][0] = crop  # 更新对比图

                # 2. 批量执行 OCR
                if pending_tasks:
                    img_paths = [task[2] for task in pending_tasks]
                    try:
                        # 写入临时文件
                        for _, img, path in pending_tasks:
                            cv2.imwrite(path, img)

                        # 调用接口
                        ocr_ret = run_fast_det_rec_ocr(img_paths, engine=global_engine)

                        # 建立 {标准路径: 文本} 的查找表
                        data_map = {os.path.normpath(item['file']): item.get('text', '')
                                    for item in ocr_ret.get('data', [])}

                        # 填回结果
                        for i, _, path in pending_tasks:
                            text = data_map.get(os.path.normpath(path), "")
                            frame_res["ocr_data"][i] = text
                            cache[i][1] = text  # 更新缓存文本

                    except Exception:
                        traceback.print_exc()
                    finally:
                        # 统一清理
                        for path in img_paths:
                            if os.path.exists(path):
                                os.remove(path)

                total_results.append(frame_res)

    finally:
        cap.release()

    return total_results


if __name__ == "__main__":
    # 测试代码保持不变
    formatted_boxes = [[[389, 914], [1049, 914], [1049, 954], [389, 954]]]
    video_file = r"W:\project\python_project\auto_video\videos\material\7459184511578852646\7459184511578852646_static_cut.mp4"
    ocr_info = [{"start": 0, "end": 5000, "boxs": formatted_boxes}]

    try:
        results = video_ocr_processor(video_file, ocr_info)
        for res in results:
            print(f"Time: {res['timestamp']:.2f}ms, Data: {res['ocr_data']}")
    except Exception as e:
        print(f"运行出错: {e}")
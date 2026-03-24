# -- coding: utf-8 --
""":authors:
    zhuxiaohu
:create_date:
    2026/3/24 22:09
:last_date:
    2026/3/24 22:09
:description:
    使用 rembg 提取图像主体，并同步分离出透明背景的主体图与挖空主体的背景图。
"""
import time
import traceback

from rembg import remove, new_session
from PIL import Image, ImageOps

def create_thumbnail_assets(input_path: str, fg_output_path: str, bg_output_path: str, model_name: str = "bria-rmbg"):
    """
    提取图片主体并保存为透明背景的 PNG，同时保存被挖空主体的背景图。
    """
    try:
        # 初始化 Session
        session_start = time.perf_counter()
        session = new_session(model_name)
        print(f"模型加载完成，耗时: {time.perf_counter() - session_start:.4f} 秒\n")

        # 读取图片，并统一转换为 RGBA 模式，为后续合并 Mask 做准备
        try:
            input_image = Image.open(input_path).convert("RGBA")
        except FileNotFoundError:
            print(f"❌ 找不到图片：{input_path}")
            return

        print("开始执行图像分割与边缘优化...")
        process_start = time.perf_counter()

        # 1. 核心处理逻辑：提取主体
        subject_image = remove(
            input_image,
            session=session,
            alpha_matting=True,
            alpha_matting_foreground_threshold=230,
            alpha_matting_background_threshold=20,
            alpha_matting_erode_size=11,
            post_process_mask=True
        )

        inference_time = time.perf_counter() - process_start
        print(f"⚡ GPU 分割推理及 Alpha Matting 耗时: {inference_time:.4f} 秒")

        # 2. 核心处理逻辑：提取背景
        print("正在分离原图背景...")
        bg_process_start = time.perf_counter()

        # 提取主体的 Alpha 通道作为 Mask
        # split() 返回 (R, G, B, A)，索引 3 即为 Alpha 通道
        subject_mask = subject_image.split()[3]

        # 反转 Mask：原本主体的白色区域变黑（透明），原本背景的黑色区域变白（保留）
        background_mask = ImageOps.invert(subject_mask)

        # 复制原图，并将反转后的 Mask 应用为原图的 Alpha 通道
        background_image = input_image.copy()
        background_image.putalpha(background_mask)

        bg_process_time = time.perf_counter() - bg_process_start
        print(f"⚡ CPU 背景分离耗时: {bg_process_time:.4f} 秒")

        # 保存结果
        subject_image.save(fg_output_path)
        background_image.save(bg_output_path)
        print(f"\n✅ 处理成功！文件已保存：")
        print(f"   🧑 主体图 (Foreground) -> {fg_output_path}")
        print(f"   🖼️ 背景图 (Background) -> {bg_output_path}")
    except Exception as e:
        traceback.print_exc()


if __name__ == "__main__":
    # 配置你的输入输出路径
    INPUT_FILE = r"W:\project\python_project\auto_video\videos\task\7619756158480410255\None\cover\7619756158480410255_00-00.jpg"

    # 自动生成主体和背景的文件路径
    FG_OUTPUT_FILE = INPUT_FILE.replace(".jpg", "_subject.png")
    BG_OUTPUT_FILE = INPUT_FILE.replace(".jpg", "_background.png")

    # 传入双路径进行处理
    create_thumbnail_assets(
        input_path=INPUT_FILE,
        fg_output_path=FG_OUTPUT_FILE,
        bg_output_path=BG_OUTPUT_FILE,
        model_name="bria-rmbg"
    )
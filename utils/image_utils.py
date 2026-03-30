import json
import os
import random
import shutil
import subprocess
from PIL import Image, ImageFont
import time
import traceback

from rembg import remove, new_session
from PIL import Image, ImageOps

from utils.common_utils import safe_process_limit

# 【优化删除】：删除了引发性能灾难的递归生成函数 get_partitions

def get_valid_intervals_sat(sat, y, fs, img_w, img_h, threshold=20):
    """
    【重构】利用二维积分图 (SAT) 极速获取可用区间，时间复杂度 O(W)
    """
    y1 = max(0, int(y))
    y2 = min(img_h, int(y + fs))
    if y2 <= y1: return []

    dy = y2 - y1
    intervals = []
    in_interval = False
    start = 0

    # 【优化】直接在安全边界内扫描，跳过两侧无用计算
    L_bound = int(img_w * 0.10)
    R_bound = int(img_w * 0.90)

    for x in range(L_bound, R_bound):
        # 核心：利用积分图 O(1) 计算 x 坐标这一列的 Alpha 总和
        col_sum = sat[y2][x + 1] - sat[y1][x + 1] - sat[y2][x] + sat[y1][x]
        avg_alpha = col_sum / dy

        if avg_alpha < threshold:
            if not in_interval:
                start = x
                in_interval = True
        else:
            if in_interval:
                intervals.append((start, x))
                in_interval = False

    if in_interval:
        intervals.append((start, R_bound))

    # 过滤过窄的区间
    safe_intervals = [(s, e) for s, e in intervals if (e - s) > fs * 0.6]
    return safe_intervals


def layout_line_fixed_gap(chars, intervals, fs):
    """
    【重构】使用动态规划(DP)替换暴力穷举，时间复杂度从 O(组合数) 降至 O(N^2 * M)
    精准寻找 "最大化-最小剩余空间 (Max-Min Slack)" 的最优排版。
    """
    n = len(chars)
    m = len(intervals)
    if n == 0 or m == 0: return None

    fixed_gap = int(fs * 0.06)

    # 1. 前缀和优化：O(1) 极速计算任意字符片段的总宽度
    prefix_w = [0] * (n + 1)
    for i in range(n):
        prefix_w[i + 1] = prefix_w[i] + chars[i][1]

    def get_group_width(start_idx, count):
        if count == 0: return 0
        return prefix_w[start_idx + count] - prefix_w[start_idx] + fixed_gap * (count - 1)

    # 2. 记忆化搜索 DP
    memo = {}

    def solve(c_idx, i_idx):
        if (c_idx, i_idx) in memo:
            return memo[(c_idx, i_idx)]

        # 边界条件：字符分完且区间遍历完 -> 合法
        if c_idx == n and i_idx == m:
            return float('inf'), []
        # 边界条件：区间用完了但字符没分完 -> 不合法
        if i_idx == m:
            return -float('inf'), None

        best_score = -float('inf')
        best_counts = None
        interval_width = intervals[i_idx][1] - intervals[i_idx][0]

        # 尝试给当前区间分配 k 个字符 (k 可以为 0)
        for k in range(n - c_idx + 1):
            group_w = get_group_width(c_idx, k)

            # 剪枝：如果当前分配的宽度已经超出区间，继续增加 k 必定也超出，直接跳出循环
            if group_w > interval_width:
                break

                # 如果分配 0 个字符，不算作挤占空间，slack 视作无限大（与原逻辑一致）
            slack = interval_width - group_w if k > 0 else float('inf')

            next_score, next_counts = solve(c_idx + k, i_idx + 1)

            if next_counts is not None:
                # 当前方案的"木桶短板"（最小剩余空间）
                current_min_slack = min(slack, next_score)
                # 寻找能让"木桶短板"最长（最均衡）的分配方案
                if current_min_slack > best_score:
                    best_score = current_min_slack
                    best_counts = [k] + next_counts

        memo[(c_idx, i_idx)] = (best_score, best_counts)
        return memo[(c_idx, i_idx)]

    max_score, optimal_counts = solve(0, 0)

    # 无解（装不下）
    if optimal_counts is None or max_score == -float('inf'):
        return None

    # 3. 根据最优解还原排版坐标
    placements = []
    char_idx = 0
    for int_idx, count in enumerate(optimal_counts):
        if count == 0: continue

        assigned_chars = chars[char_idx: char_idx + count]
        group_width = get_group_width(char_idx, count)
        start_x, end_x = intervals[int_idx]

        slack = (end_x - start_x) - group_width
        x = start_x + slack / 2  # 居中对齐

        for c, w in assigned_chars:
            placements.append({'char': c, 'x': x})
            x += w + fixed_gap

        char_idx += count

    return placements


def try_layout_rigid_block(lines, fs, sat, img_w, img_h, font_path, spacing_ratio, position):
    """
    【重构】恢复原始代码精准的 Y 轴高度逻辑，摒弃上下扫描
    """
    font = ImageFont.truetype(font_path, int(fs))

    lines_chars = []
    for line in lines:
        chars = [(c, font.getlength(c)) for c in line if c.strip() != '']
        if chars:
            lines_chars.append(chars)

    if not lines_chars: return False, None

    # ===== 核心修复：完全还原旧代码的高度计算逻辑 =====
    true_high = int(img_w * 9 / 16)
    line_height = int(fs * spacing_ratio)
    total_text_height = line_height * (len(lines_chars) - 1) + fs

    position_map = {
        'center': img_h / 2,
        'top_third': (img_h / 2 - true_high / 2 + fs / 2),
        'bottom_third': img_h * 0.75
    }
    block_y_center = position_map.get(position, img_h * 0.5)
    start_y = block_y_center - total_text_height / 2
    # ===============================================

    block_success = True
    layout_result = []

    # 直接使用计算好的完美的 start_y 进行排版测试
    for i, chars in enumerate(lines_chars):
        current_line_y = start_y + i * line_height

        # 安全校验：如果高度直接超出图片外则判定该字号失败
        if current_line_y < 0 or current_line_y + fs > img_h:
            return False, None

        intervals = get_valid_intervals_sat(sat, current_line_y, fs, img_w, img_h)
        if not intervals:
            block_success = False
            break

        placements = layout_line_fixed_gap(chars, intervals, fs)
        if placements is None:
            block_success = False
            break

        layout_result.append({'y': current_line_y, 'placements': placements})

    if block_success:
        return True, layout_result

    return False, None


def create_enhanced_cover_layered(
        bg_image_path: str,
        fg_image_path: str,
        output_image_path: str,
        text_lines: list[str],
        font_path='C:/Windows/Fonts/msyhbd.ttc',
        position: str = 'top_third',  # ⬅️ 核心修复：请回原版的 position 参数
        color_theme: str = 'auto',
        font_size_ratio: float = 1.0,
        line_spacing_ratio: float = 1.3,
        overwrite: bool = True
) -> str or None:
    if not all([os.path.exists(bg_image_path), os.path.exists(fg_image_path), os.path.exists(font_path)]):
        print("错误: 文件未找到。")
        return None

    with Image.open(bg_image_path) as bg_img:
        img_w, img_h = bg_img.size
    with Image.open(fg_image_path) as fg_img:
        fg_alpha = fg_img.convert("RGBA").split()[3]

    color_themes = {
        # 1. 绝对主力 (万能底牌，最高清晰度：黑底/黑边/黑阴影压阵)
        'classic_white': {'fontcolor': 'white', 'bordercolor': 'black', 'shadowcolor': 'black@0.8'},
        'vibrant_yellow': {'fontcolor': '#FFD700', 'bordercolor': 'black', 'shadowcolor': 'black@0.8'},

        # 2. 情绪与警告 (高饱和度+白边反差：红/粉等颜色必须加白边才能在暗色或花哨背景中跳脱出来)
        'alert_red': {'fontcolor': '#FF0000', 'bordercolor': 'white', 'shadowcolor': 'black@0.6'},
        'toxic_magenta': {'fontcolor': '#FF00FF', 'bordercolor': 'white', 'shadowcolor': 'black@0.6'},
        'brand_orange': {'fontcolor': '#FF6600', 'bordercolor': 'white', 'shadowcolor': 'black@0.6'},

        # 3. 科技与潮流 (冷色调强对比：适合科普、数码)
        'success_green': {'fontcolor': '#00FF00', 'bordercolor': 'black', 'shadowcolor': 'black@0.8'},
        'cyber_cyan': {'fontcolor': '#00FFFF', 'bordercolor': 'black', 'shadowcolor': 'black@0.8'},

        # 4. 极端亮度兜底 (亮色/纯白背景专用：极黑字+纯白边，去除阴影防止画面显脏)
        'dark_inverse': {'fontcolor': '#111111', 'bordercolor': 'white', 'shadowcolor': 'black@0.0'},
    }

    # 修复 Auto 逻辑：优先使用最稳妥的黄/白，而不是全局随机
    if color_theme not in color_themes or color_theme == 'auto':
        chosen_theme = color_themes[random.choice(['classic_white', 'vibrant_yellow'])]
    else:
        chosen_theme = color_themes[color_theme]

    # ================= 核心性能优化 =================
    # print("\n>>> [Pass 0] 预处理遮罩: 生成 Alpha 通道积分图 (SAT)...")
    alpha_data = list(fg_alpha.getdata())
    sat = [[0] * (img_w + 1) for _ in range(img_h + 1)]
    idx = 0
    for y in range(1, img_h + 1):
        row_sum = 0
        for x in range(1, img_w + 1):
            row_sum += alpha_data[idx]
            sat[y][x] = sat[y - 1][x] + row_sum
            idx += 1
    # ===============================================

    # print(">>> [Pass 1] 启动【刚性区块】顶格排版，计算绝对完美字号...")

    clean_lines = [line for line in text_lines if line.strip()]
    if not clean_lines: return None

    longest_line = max(clean_lines, key=len)
    base_target_fs = int(min((img_w * 0.8 / max(1, len(longest_line))), img_h / 4) * font_size_ratio)

    min_fs = int(img_w * 0.04)
    max_fs = int(base_target_fs * 1.5)

    best_fs = min_fs
    best_layout = None

    low, high = min_fs, max_fs
    while low <= high:
        mid_fs = (low + high) // 2
        # 将 position 传递进去
        success, layout = try_layout_rigid_block(clean_lines, mid_fs, sat, img_w, img_h, font_path,
                                                 line_spacing_ratio, position)
        if success:
            best_fs = mid_fs
            best_layout = layout
            low = mid_fs + 1
        else:
            high = mid_fs - 1

    if not best_layout:
        raise RuntimeError("无法找到合适的字号进行刚性排版，理论上不应该发生。")
        print("⚠️ 警告: 画面极度拥挤，触发刚性居中兜底排版。")
        fallback_intervals = [(int(img_w * 0.10), int(img_w * 0.90))]
        best_fs = min_fs
        font = ImageFont.truetype(font_path, best_fs)

        # ===== 兜底排版也恢复原版高度计算 =====
        true_high = int(img_w * 9 / 16)
        total_text_height = int(best_fs * line_spacing_ratio) * (len(clean_lines) - 1) + best_fs

        position_map = {
            'center': img_h / 2,
            'top_third': (img_h / 2 - true_high / 2 + best_fs / 2),
            'bottom_third': img_h * 0.75
        }
        block_y_center = position_map.get(position, img_h * 0.5)
        block_start_y = block_y_center - total_text_height / 2
        # ===================================

        best_layout = []
        for i, line in enumerate(clean_lines):
            chars = [(c, font.getlength(c)) for c in line if c.strip() != '']
            placements = layout_line_fixed_gap(chars, fallback_intervals, best_fs)

            current_line_y = block_start_y + i * int(best_fs * line_spacing_ratio)
            best_layout.append({'y': current_line_y, 'placements': placements})

    # print(f"✅ 计算完毕！敲定统一字号: {best_fs}px (字距/行距已完全数学锁定)")

    escaped_font_path = font_path.replace(':', '\\:') if os.name == 'nt' else font_path

    # !! 关键修改：增加描边宽度计算，配合阴影打造极强可读性 !!
    border_width = max(2, int(best_fs * 0.04))
    shadow_offset = max(2, int(best_fs * 0.06))

    drawtexts = []
    if best_layout:
        for i, line_info in enumerate(best_layout):
            if not line_info['placements']: continue
            placements = line_info['placements']

            # print(f"  - 刚性块 第 {i + 1} 行 -> Y轴: {int(line_info['y'])}, X轴跨度: {int(placements[0]['x'])} 至 {int(placements[-1]['x'])}")

            for p in placements:
                char = p['char'].replace(':', '\\:').replace('%', '\\%').replace('\'', '')
                opts = {
                    'fontfile': f"'{escaped_font_path}'",
                    'text': f"'{char}'",
                    'fontsize': str(best_fs),
                    'fontcolor': chosen_theme['fontcolor'],
                    'x': str(int(p['x'])),
                    'y': str(int(line_info['y'])),
                    'borderw': str(border_width),  # <-- 引入描边宽度
                    'bordercolor': chosen_theme['bordercolor'],  # <-- 引入描边颜色
                    'shadowcolor': chosen_theme['shadowcolor'],
                    'shadowx': str(shadow_offset),
                    'shadowy': str(shadow_offset)
                }
                drawtexts.append("drawtext=" + ":".join(f"{k}={v}" for k, v in opts.items()))

    drawtext_chain = ",".join(drawtexts)
    filter_complex = f"[0:v]{drawtext_chain}[bg_text];[bg_text][1:v]overlay=0:0"

    command = ['ffmpeg', '-i', bg_image_path, '-i', fg_image_path, '-filter_complex', filter_complex]
    if overwrite: command.append('-y')
    command.append(output_image_path)

    try:
        subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
        print(f"🎉 成功! (终极刚性排版) 封面已保存到 '{output_image_path}'")
        return output_image_path
    except subprocess.CalledProcessError as e:
        print("FFMPEG 执行失败!")
        print(e.stderr)
        return None





@safe_process_limit(limit=4, name="create_thumbnail_assets")
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

        # print("开始执行图像分割与边缘优化...")
        process_start = time.perf_counter()

        # 1. 核心处理逻辑：提取主体
        subject_image = remove(
            input_image,
            session=session,
            # 关闭所有的 alpha_matting 和 post_process 干扰
            alpha_matting=False,
            post_process_mask=False
        )
        inference_time = time.perf_counter() - process_start
        print(f"⚡ GPU 分割推理及 Alpha Matting 耗时: {inference_time:.4f} 秒")

        # 2. 核心处理逻辑：提取背景
        # print("正在分离原图背景...")
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
        # print(f"\n✅ 处理成功！文件已保存：")
        # print(f"   🧑 主体图 (Foreground) -> {fg_output_path}")
        # print(f"   🖼️ 背景图 (Background) -> {bg_output_path}")
    except Exception as e:
        traceback.print_exc()

def _get_image_dimensions(image_path: str) -> tuple[int, int] or None:
    # (此辅助函数无需修改)
    command = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height', '-of', 'json', image_path
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
        data = json.loads(result.stdout)
        return data['streams'][0]['width'], data['streams'][0]['height']
    except Exception as e:
        print(f"错误: 无法获取图片尺寸 '{image_path}'.")
        print(f"具体错误: {e}")
        return None



def create_enhanced_cover(
        input_image_path: str,
        output_image_path: str,
        text_lines: list[str],
        font_path='C:/Windows/Fonts/msyhbd.ttc',
        position: str = 'top_third',
        color_theme: str = 'auto',
        font_size_ratio: float = 1.0,
        line_spacing_ratio: float = 1.4,
        overwrite: bool = True
) -> str or None:
    if not all([os.path.exists(input_image_path), os.path.exists(font_path)]):
        print("错误: 输入文件或字体文件未找到。")
        return None

    dimensions = _get_image_dimensions(input_image_path)
    if not dimensions: return None
    img_w, img_h = dimensions
    true_high = int(img_w * 9 / 16)

    if not text_lines:
        print("警告: 未提供任何文字，将直接复制图片。")
        if overwrite or not os.path.exists(output_image_path):
            shutil.copy(input_image_path, output_image_path)
        return output_image_path

    color_themes = {
        # 1. 绝对主力 (万能底牌，最高清晰度：黑底/黑边/黑阴影压阵)
        'classic_white': {'fontcolor': 'white', 'bordercolor': 'black', 'shadowcolor': 'black@0.8'},
        'vibrant_yellow': {'fontcolor': '#FFD700', 'bordercolor': 'black', 'shadowcolor': 'black@0.8'},

        # 2. 情绪与警告 (高饱和度+白边反差：红/粉等颜色必须加白边才能在暗色或花哨背景中跳脱出来)
        'alert_red': {'fontcolor': '#FF0000', 'bordercolor': 'white', 'shadowcolor': 'black@0.6'},
        'toxic_magenta': {'fontcolor': '#FF00FF', 'bordercolor': 'white', 'shadowcolor': 'black@0.6'},
        'brand_orange': {'fontcolor': '#FF6600', 'bordercolor': 'white', 'shadowcolor': 'black@0.6'},

        # 3. 科技与潮流 (冷色调强对比：适合科普、数码)
        'success_green': {'fontcolor': '#00FF00', 'bordercolor': 'black', 'shadowcolor': 'black@0.8'},
        'cyber_cyan': {'fontcolor': '#00FFFF', 'bordercolor': 'black', 'shadowcolor': 'black@0.8'},

        # 4. 极端亮度兜底 (亮色/纯白背景专用：极黑字+纯白边，去除阴影防止画面显脏)
        'dark_inverse': {'fontcolor': '#111111', 'bordercolor': 'white', 'shadowcolor': 'black@0.0'},    }
    # 如果指定的主题不存在，或为 'auto'，则从预设中随机选择
    if color_theme not in color_themes or color_theme == 'auto':
        # 默认随机选择，但可以优先选择最经典的
        # chosen_theme = color_themes['classic_white']
        chosen_theme = random.choice(list(color_themes.values()))
    else:
        chosen_theme = color_themes[color_theme]

    longest_line = max(text_lines, key=len)
    longest_line_size = max(8, len(longest_line))  # 避免极端短文本导致字体过大
    target_text_width = img_w * 0.9
    estimated_char_width_ratio = 1.0
    font_size = int(min((target_text_width / longest_line_size), img_h / 4) * font_size_ratio)

    # !! 关键修改 2: 增加阴影偏移量，模拟更厚的描边效果 !!
    # 将偏移量从原来的5%提升到8%
    shadow_offset = max(2, int(font_size * 0.06))

    line_height = int(font_size * line_spacing_ratio)
    total_text_height = line_height * (len(text_lines) - 1) + font_size

    escaped_font_path = font_path.replace(':', '\\:') if os.name == 'nt' else font_path

    position_map = {'center': img_h / 2, 'top_third': (img_h / 2 - true_high / 2 + font_size / 2),
                    'bottom_third': img_h * 0.75}
    block_y_center = position_map.get(position, img_h * 0.5)  # 默认居中
    start_y = block_y_center - total_text_height / 2

    filters = []
    for i, line in enumerate(text_lines):
        line_y = start_y + i * line_height
        x_expr = '(w-text_w)/2'

        drawtext_options = {
            'fontfile': f"'{escaped_font_path}'",
            'text': f"'{line.replace(':', '\\:').replace('%', '\\%').replace('\'', '')}'",
            'fontsize': str(font_size),
            'fontcolor': chosen_theme['fontcolor'],
            'x': x_expr,
            'y': str(line_y),
            'shadowcolor': chosen_theme['shadowcolor'],
            'shadowx': str(shadow_offset),
            'shadowy': str(shadow_offset)
        }
        filters.append("drawtext=" + ":".join(f"{k}={v}" for k, v in drawtext_options.items()))

    vf_string = ",".join(filters)
    command = ['ffmpeg', '-i', input_image_path, '-vf', vf_string]
    if overwrite: command.append('-y')
    command.append(output_image_path)

    print(f"主题: {chosen_theme}")

    try:
        subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
        print(f"🎉 成功! 优化后的封面已保存到 '{output_image_path}'")
        return output_image_path
    except subprocess.CalledProcessError as e:
        print("FFMPEG 执行失败!")
        print(f"错误码: {e.returncode}")
        print("FFMPEG 输出 (stderr):")
        print(e.stderr)
        return None

def create_enhanced_cover_auto(
        input_image_path: str,
        output_image_path: str,
        text_lines: list[str],
        font_path='C:/Windows/Fonts/msyhbd.ttc',
        position: str = 'top_third',
        color_theme: str = 'auto',
        font_size_ratio: float = 1.0,
        line_spacing_ratio: float = 1.4,
        overwrite: bool = True
) -> str or None:
    """
    智能封面生成函数（带兜底机制）：
    优先尝试使用 create_enhanced_cover_layered（三明治防遮挡模式），
    如果抠图分离失败或新排版算法报错，则自动回退到 create_enhanced_cover（经典单图覆盖模式）。
    """
    if not os.path.exists(input_image_path):
        print(f"错误: 找不到输入文件 '{input_image_path}'。")
        return None

    # 准备临时文件的路径
    base_path, ext = os.path.splitext(input_image_path)
    temp_fg_path = f"{base_path}_auto_temp_fg.png"
    temp_bg_path = f"{base_path}_auto_temp_bg.png"

    layered_success = False
    result_path = None

    try:
        print("\n>>> [Auto Mode] 尝试使用新版分层模式 (Layered) 生成封面...")
        if not os.path.exists(temp_fg_path) or not os.path.exists(temp_bg_path):
            # 1. 自动调用抠图分离背景和主体，生成临时文件
            create_thumbnail_assets(
                input_path=input_image_path,
                fg_output_path=temp_fg_path,
                bg_output_path=temp_bg_path
            )

        # 检查文件是否成功生成
        if os.path.exists(temp_fg_path) and os.path.exists(temp_bg_path):
            # 2. 调用新版分层排版
            result_path = create_enhanced_cover_layered(
                bg_image_path=temp_bg_path,
                fg_image_path=temp_fg_path,
                output_image_path=output_image_path,
                text_lines=text_lines,
                font_path=font_path,
                position=position,
                color_theme=color_theme,
                font_size_ratio=font_size_ratio,
                line_spacing_ratio=line_spacing_ratio,
                overwrite=overwrite
            )

            if result_path:
                layered_success = True
        else:
            print("⚠️ 主体或背景分离失败，找不到临时文件，准备降级处理。")

    except Exception as e:
        print(f"⚠️ 分层模式处理过程中发生异常: {e}")
        traceback.print_exc()


    # 3. 兜底逻辑：如果新版失败（未成功生成 layered_success），调用老版本
    if not layered_success:
        print("\n>>> [Auto Mode fallback] 触发兜底机制，切换至经典模式 (Classic) 生成封面...")
        result_path = create_enhanced_cover(
            input_image_path=input_image_path,
            output_image_path=output_image_path,
            text_lines=text_lines,
            font_path=font_path,
            position=position,
            color_theme=color_theme,
            font_size_ratio=font_size_ratio,
            line_spacing_ratio=line_spacing_ratio,
            overwrite=overwrite
        )

    return result_path


if __name__ == "__main__":
    # # 配置你的输入输出路径
    # INPUT_FILE = r"W:\project\python_project\auto_video\videos\task\7622116077577956963\69c919a5d48b8dfd61523d00\cover\7622116077577956963.jpg"
    #
    # # 自动生成主体和背景的文件路径
    # FG_OUTPUT_FILE = INPUT_FILE.replace(".jpg", "_subject.png")
    # BG_OUTPUT_FILE = INPUT_FILE.replace(".jpg", "_background.png")
    #
    # # 传入双路径进行处理
    # create_thumbnail_assets(
    #     input_path=INPUT_FILE,
    #     fg_output_path=FG_OUTPUT_FILE,
    #     bg_output_path=BG_OUTPUT_FILE,
    #     model_name="bria-rmbg"
    # )


    # ==========================================
    # 测试数据准备
    # ==========================================
    # 假设我们要写入的封面文案
    texts = ["AI深度解析", "核心原理解密"]

    # 你的字体路径 (Windows默认微软雅黑粗体)
    font = "C:/Windows/Fonts/msyhbd.ttc"

    # ==========================================
    # 场景 1：调用老函数 (经典单图覆盖模式)
    # ==========================================
    print(">>> 开始生成老版本封面 (文字在最上层)...")

    # 老版本只需要一张完整的底图
    old_input_img = r"W:\project\python_project\auto_video\videos\task\7622116077577956963\69c919a5d48b8dfd61523d00\cover\7622116077577956963.jpg"
    old_output_img = old_input_img.replace(".jpg", "_old.png")  # 输出为 PNG 保持质量

    create_enhanced_cover(
        input_image_path=old_input_img,
        output_image_path=old_output_img,
        text_lines=texts,
        font_path=font,
        position='top_third',  # 期望位置在上方三分之一处
        color_theme='vibrant_yellow',  # 爆款黄色主题
        font_size_ratio=1.0,
        overwrite=True
    )

    # ==========================================
    # 场景 2：调用新函数 (三明治防遮挡模式)
    # ==========================================
    print("\n>>> 开始生成新版本封面 (背景-文字-主体 三层结构)...")

    # 新版本需要你提前把 test_input.jpg 拆分成两张图：
    # 1. test_bg.png: 纯背景图
    # 2. test_fg.png: 抠好的主体图 (必须带有Alpha透明通道，也就是抠图后背景是透明的PNG)
    new_bg_img = r"W:\project\python_project\auto_video\videos\task\7622116077577956963\69c919a5d48b8dfd61523d00\cover\7622116077577956963_background.png"
    new_fg_img = r"W:\project\python_project\auto_video\videos\task\7622116077577956963\69c919a5d48b8dfd61523d00\cover\7622116077577956963_subject.png"
    new_output_img = new_fg_img.replace("_subject.png", "_layered.png")  # 输出为 PNG 保持质量

    create_enhanced_cover_layered(
        bg_image_path=new_bg_img,  # 传参变化 1：传入背景
        fg_image_path=new_fg_img,  # 传参变化 2：传入透明主体
        output_image_path=new_output_img,
        text_lines=texts,
        font_path=font,
        color_theme='vibrant_yellow',
        font_size_ratio=1.0,
        overwrite=True
    )
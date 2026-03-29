import os
import random
import subprocess
from PIL import Image, ImageFont


def get_partitions(n, m):
    if m == 1:
        yield (n,)
        return
    for i in range(n + 1):
        for p in get_partitions(n - i, m - 1):
            yield (i,) + p


def get_valid_intervals(fg_alpha, y, fs, img_w):
    band = fg_alpha.crop((0, int(y), int(img_w), int(y + fs)))
    squashed = band.resize((int(img_w), 1), resample=Image.Resampling.BOX)
    alphas = list(squashed.getdata())

    threshold = 20
    intervals = []
    in_interval = False
    start = 0
    for x, alpha in enumerate(alphas):
        if alpha < threshold:
            if not in_interval:
                start = x
                in_interval = True
        else:
            if in_interval:
                intervals.append((start, x))
                in_interval = False
    if in_interval: intervals.append((start, int(img_w)))

    L_bound = int(img_w * 0.10)
    R_bound = int(img_w * 0.90)

    safe_intervals = []
    for s, e in intervals:
        s_clipped = max(s, L_bound)
        e_clipped = min(e, R_bound)
        if e_clipped - s_clipped > fs * 0.6:
            safe_intervals.append((s_clipped, e_clipped))
    return safe_intervals


def layout_line_fixed_gap(chars, intervals, fs):
    n = len(chars)
    m = len(intervals)
    if n == 0 or m == 0: return None

    # 【严格字距】: 固定为字体大小的 6%
    fixed_gap = int(fs * 0.06)

    best_partition = None
    best_placements = None
    max_score = -999999

    for partition in get_partitions(n, m):
        valid = True
        placements = []
        char_idx = 0
        min_slack = 999999

        for int_idx, count in enumerate(partition):
            if count == 0: continue
            assigned_chars = chars[char_idx: char_idx + count]
            sum_w = sum(w for c, w in assigned_chars)

            group_width = sum_w + fixed_gap * (count - 1)
            start_x, end_x = intervals[int_idx]
            interval_width = end_x - start_x

            if group_width > interval_width:
                valid = False
                break

            slack = interval_width - group_width
            if slack < min_slack: min_slack = slack

            x = start_x + slack / 2
            for c, w in assigned_chars:
                placements.append({'char': c, 'x': x})
                x += w + fixed_gap
            char_idx += count

        if valid and min_slack > max_score:
            max_score = min_slack
            best_partition = partition
            best_placements = placements

    return best_placements


def try_layout_rigid_block(lines, fs, fg_alpha, img_w, img_h, font_path, spacing_ratio):
    """
    【核心重构】将所有行作为一个“刚性硬块”进行整体排版测试
    """
    font = ImageFont.truetype(font_path, int(fs))
    true_high = int(img_w * 9 / 16)
    valid_y_min = int((img_h - true_high) / 2)
    valid_y_max = int((img_h + true_high) / 2)

    # 计算刚性文字块的总高度
    total_block_height = int(fs + (len(lines) - 1) * (fs * spacing_ratio))

    # 顶格起步探测点
    search_start = valid_y_min + int(fs * 0.1)
    search_end = valid_y_max - total_block_height

    if search_end < search_start:
        search_end = search_start  # 至少测一次

    # 将整个刚性文字块在画面中平移扫描
    for block_y in range(int(search_start), int(search_end) + 1, 15):
        block_success = True
        layout_result = []

        for i, line in enumerate(lines):
            chars = [(c, font.getlength(c)) for c in line if c.strip() != '']
            if not chars: continue

            # 【严格行距】: Y坐标完全由数学公式锁死，绝不允许独立偏移！
            current_line_y = block_y + i * int(fs * spacing_ratio)

            intervals = get_valid_intervals(fg_alpha, current_line_y, fs, img_w)
            if not intervals:
                block_success = False
                break  # 只要有一行在这个高度被挡死，整个区块就宣告失败

            placements = layout_line_fixed_gap(chars, intervals, fs)
            if placements is None:
                block_success = False
                break  # 塞不进去，宣告失败

            layout_result.append({'y': current_line_y, 'placements': placements})

        if block_success:
            return True, layout_result  # 找到了完美的区块落脚点！

    return False, None


def create_enhanced_cover_layered(
        bg_image_path: str,
        fg_image_path: str,
        output_image_path: str,
        text_lines: list[str],
        font_path='C:/Windows/Fonts/msyhbd.ttc',
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
        'vibrant_yellow': {'fontcolor': '#FFD700', 'shadowcolor': 'black@0.9'},
        'classic_white': {'fontcolor': 'White', 'shadowcolor': 'black@0.9'},
        'alert_red': {'fontcolor': '#FF2400', 'shadowcolor': 'black@0.85'},
        'cyber_cyan': {'fontcolor': '#00FFFF', 'shadowcolor': 'black@0.8'},
    }
    chosen_theme = color_themes.get(color_theme, random.choice(list(color_themes.values())))
    if color_theme == 'auto': chosen_theme = random.choice(list(color_themes.values()))

    print("\n>>> [Pass 1] 启动【刚性区块】顶格排版，计算绝对完美字号...")

    # 剔除空行，保证刚性区块的连续性
    clean_lines = [line for line in text_lines if line.strip()]
    if not clean_lines: return None

    longest_line = max(clean_lines, key=len)
    base_target_fs = int(min((img_w * 0.8 / max(1, len(longest_line))), img_h / 4) * font_size_ratio)

    min_fs = int(img_w * 0.04)
    max_fs = int(base_target_fs * 1.5)

    best_fs = min_fs
    best_layout = None

    # 二分查找：这不仅仅是在找字号，是在找一个【能被刚性塞入】的最大字号
    low, high = min_fs, max_fs
    while low <= high:
        mid_fs = (low + high) // 2
        success, layout = try_layout_rigid_block(clean_lines, mid_fs, fg_alpha, img_w, img_h, font_path,
                                                 line_spacing_ratio)
        if success:
            best_fs = mid_fs
            best_layout = layout
            low = mid_fs + 1
        else:
            high = mid_fs - 1

            # 兜底硬排
    if not best_layout:
        print("⚠️ 警告: 画面极度拥挤，触发刚性居中兜底排版。")
        fallback_intervals = [(int(img_w * 0.10), int(img_w * 0.90))]
        best_fs = min_fs
        font = ImageFont.truetype(font_path, best_fs)

        true_high = int(img_w * 9 / 16)
        block_start_y = int((img_h - true_high) / 2) + int(best_fs * 0.1)

        best_layout = []
        for i, line in enumerate(clean_lines):
            chars = [(c, font.getlength(c)) for c in line if c.strip() != '']
            placements = layout_line_fixed_gap(chars, fallback_intervals, best_fs)

            # 行距依然被严格锁死
            current_line_y = block_start_y + i * int(best_fs * line_spacing_ratio)
            best_layout.append({'y': current_line_y, 'placements': placements})

    print(f"✅ 计算完毕！敲定统一字号: {best_fs}px (字距/行距已完全数学锁定)")

    escaped_font_path = font_path.replace(':', '\\:') if os.name == 'nt' else font_path
    shadow_offset = max(2, int(best_fs * 0.06))

    drawtexts = []
    if best_layout:
        for i, line_info in enumerate(best_layout):
            if not line_info['placements']: continue
            placements = line_info['placements']

            print(
                f"  - 刚性块 第 {i + 1} 行 -> Y轴: {int(line_info['y'])}, X轴跨度: {int(placements[0]['x'])} 至 {int(placements[-1]['x'])}")

            for p in placements:
                char = p['char'].replace(':', '\\:').replace('%', '\\%').replace('\'', '')
                opts = {
                    'fontfile': f"'{escaped_font_path}'",
                    'text': f"'{char}'",
                    'fontsize': str(best_fs),
                    'fontcolor': chosen_theme['fontcolor'],
                    'x': str(int(p['x'])),
                    'y': str(int(line_info['y'])),
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
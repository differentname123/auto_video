import time

import requests
import json

from utils.common_utils import read_json, save_json


def send_generate_request(video_id1, video_id2, user_name='dahao'):
    """
    向指定接口发送生成视频的请求。

    Args:
        video_id1 (str): 第一个视频的ID
        video_id2 (str): 第二个视频的ID

    Returns:
        dict: 接口返回的JSON响应，如果请求失败返回None
    """
    url = "http://127.0.0.1:5002/one-click-generate"

    # 构造请求头
    headers = {
        "Content-Type": "application/json"
    }

    # 构造请求体 (Payload)
    # 注意：这里直接使用了Python的True/False，requests库在发送JSON时会自动转换为json格式的true/false
    payload = {
        'userName': user_name,
        'global_settings': {
            'video_type': '通用',
            'retention_ratio': 'free',
            'is_need_audio_replace': True,
            'is_need_scene_title': True,
            'is_need_commentary': True,
            'schedule_date': '2026-01-05',
            'creative_guidance': ''
        },
        'video_list': [
            {
                '_cardDomId': 'video-card-1',
                # 使用 f-string 替换 video_id1
                'original_url': f'https://www.douyin.com/video/{video_id1}',
                'is_contains_author_voice': True,
                'is_contains_bgm': True,
                'is_realtime_video': True,
                'is_requires_text': True,
                'is_needs_stickers': True,
                'max_scenes': 0,
                'has_ad_or_face': 'auto',
                'scene_order_fixed': 'auto',
                'split_guidance': '',
                'remove_time_segments': [],
                'split_time_points': []
            },
            {
                '_cardDomId': 'video-card-2',
                # 使用 f-string 替换 video_id2
                'original_url': f'https://www.douyin.com/video/{video_id2}',
                'is_contains_author_voice': True,
                'is_contains_bgm': True,
                'is_realtime_video': True,
                'is_requires_text': True,
                'is_needs_stickers': True,
                'max_scenes': 0,
                'has_ad_or_face': 'auto',
                'scene_order_fixed': 'auto',
                'split_guidance': '',
                'remove_time_segments': [],
                'split_time_points': []
            }
        ]
    }

    try:
        # 发送 POST 请求
        response = requests.post(url, json=payload, headers=headers)

        # 检查响应状态码
        response.raise_for_status()

        # 返回解析后的JSON数据
        print(f"请求发送成功，状态码: {response.status_code}")
        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"请求发送失败: {e}")
    finally:
        # 如果有响应内容（比如 400 错误通常会带回错误原因），在这里打印
        if response is not None:
            try:
                # 尝试解析 JSON 并以中文显示
                error_json = response.json()
                print("服务器返回的错误详情:")
                print(json.dumps(error_json, ensure_ascii=False, indent=2))
            except json.JSONDecodeError:
                # 如果不是 JSON，直接打印文本
                print(response.text)
        return None


# --- 使用示例 ---
if __name__ == "__main__":
    video_content_plans_file = r'W:\project\python_project\watermark_remove\LLM\TikTokDownloader\back_up\video_content_plans_similar_videos.json'
    video_play_comment_file = r'W:\project\python_project\watermark_remove\LLM\TikTokDownloader\back_up\video_play_comment.json'
    used_video_file = r'W:\project\python_project\auto_video\config\used_video.json'
    used_video_list = read_json(used_video_file)
    plans_video = read_json(video_content_plans_file)
    video_play_comment_info = read_json(video_play_comment_file)

    for key, info in plans_video.items():
        video_id_list = key.split('_')
        score = info.get('score', 0)
        for video_id in video_id_list:
            other_score = video_play_comment_info.get(video_id, {}).get('score', 0)
            comment = video_play_comment_info.get(video_id, {}).get('comment', 0)
            play = video_play_comment_info.get(video_id, {}).get('play', 0)
            if other_score > score:
                score = other_score
                info['comment'] = comment
                info['play'] = play
        info['score'] = score

    # 将plans_video按照score降序排序
    sorted_videos = sorted(plans_video.items(), key=lambda x: x[1].get('score', 0), reverse=True)

    for sorted_video in sorted_videos:
        video_key = sorted_video[0]
        if video_key in used_video_list:
            print(f"视频对 {video_key} 已使用，跳过。")
            continue
        used_video_list.append(video_key)
        value = sorted_video[1]
        video_keys = value.get('video_keys', [])
        id_1 = video_keys[0]
        id_2 = video_keys[1]
        print(f"当前时间 {time.strftime('%Y-%m-%d %H:%M:%S')} 正在处理视频对: {id_1} 和 {id_2}，分数: {value.get('score', 0)}")
        result = send_generate_request(id_1, id_2)
        save_json(used_video_file, used_video_list)

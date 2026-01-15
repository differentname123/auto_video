import random
import time
import traceback
from collections import defaultdict

import requests
import json

from application.video_common_config import USER_STATISTIC_INFO_PATH, STATISTIC_PLAY_COUNT_FILE
from utils.common_utils import read_json, save_json, get_user_type
from utils.mongo_base import gen_db_object
from utils.mongo_manager import MongoManager


def send_generate_request(video_id1, video_id2, user_name='dahao'):
    """
    向指定接口发送生成视频的请求。

    Args:
        video_id1 (str): 第一个视频的ID
        video_id2 (str): 第二个视频的ID

    Returns:
        dict: 接口返回的JSON响应，如果请求失败返回None
    """
    url = "http://127.0.0.1:5001/one-click-generate"

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

def auto_send():
    video_content_plans_file = r'W:\project\python_project\watermark_remove\LLM\TikTokDownloader\back_up\video_content_plans_similar_videos.json'
    video_play_comment_file = r'W:\project\python_project\watermark_remove\LLM\TikTokDownloader\back_up\video_play_comment.json'
    used_video_file = r'W:\project\python_project\auto_video\config\used_video.json'
    used_video_list = read_json(used_video_file)
    plans_video = read_json(video_content_plans_file)
    video_play_comment_info = read_json(video_play_comment_file)
    user_statistic_info = read_json(USER_STATISTIC_INFO_PATH)
    user_config = read_json(r'W:\project\python_project\auto_video\config\user_config.json')

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

    need_process_users = ['lin', 'dahao', 'zhong', 'ping', "qizhu", 'mama', 'hong']
    user_type_info = user_config.get('user_type_info', {})
    select_info = {}
    for user_name in need_process_users:
        total_count = user_statistic_info.get(user_name, {}).get('today_process', 0)
        target_count = 30
        need_count = max(target_count - total_count, 0)
        preferred_video_type= 'fun'
        for video_type, user_list in user_type_info.items():
            if user_name in user_list:
                preferred_video_type = video_type
                break
        print(f"用户 {user_name} 今日已收到 {total_count} 个任务，还需处理 {need_count} 个。才能够达到目标 {target_count} 个。题材{preferred_video_type}")
        count = 0
        for sorted_video in sorted_videos:
            video_key = sorted_video[0]
            if video_key in used_video_list or video_key in select_info:
                # print(f"视频对 {video_key} 已使用，跳过。")
                continue
            value = sorted_video[1]
            video_type_cn = value.get('video_type', '娱乐')
            video_type_en = 'fun'
            if video_type_cn == '娱乐':
                video_type_en = 'fun'
            if video_type_cn == '游戏':
                video_type_en = 'game'
            if video_type_cn == '体育':
                video_type_en = 'sport'
            if video_type_en == preferred_video_type:
                value['user_name'] = user_name
                select_info[video_key] = value
                count += 1
                if count >= need_count:
                    break
        print(f"用户 {user_name} 选择了 {count} 个视频对用于处理。当前总共选择了 {len(select_info)} 个视频对。")
    print(f"总共选择了 {len(select_info)} 个视频对用于处理。")

    for video_key, value in select_info.items():
        used_video_list.append(video_key)
        user_name = value.get('user_name', '未知')
        video_keys = value.get('video_keys', [])
        id_1 = video_keys[0]
        id_2 = video_keys[1]
        print(f"{user_name} 当前时间 {time.strftime('%Y-%m-%d %H:%M:%S')} 正在处理视频对: {id_1} 和 {id_2}，分数: {value.get('score', 0)}")
        result = send_generate_request(id_1, id_2, user_name=user_name)
        save_json(used_video_file, used_video_list)

def send_good_video_quest(payload):
    url = "http://127.0.0.1:5002/one-click-generate"

    # 构造请求头
    headers = {
        "Content-Type": "application/json"
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

def send_good_video():
    # mongo_base_instance = gen_db_object()
    # manager = MongoManager(mongo_base_instance)
    # # 获取当前时间戳
    # current_timestamp = int(time.time())
    # # 往前推2天的时间戳
    # pre_timestamp = current_timestamp - 2 * 24 * 60 * 60
    # query_2 = {
    #     "created": {"$gt": pre_timestamp},
    #     "status": "已投稿"
    # }
    # # all_task = manager.find_by_custom_query(manager.tasks_collection, query_2)
    need_process_users = ['lin', 'dahao', 'zhong', 'ping', "qizhu", 'mama', 'hong', 'xiaosu', 'jie', 'qiqixiao', 'yang', 'xue']
    simple_need_process_users = ['yang', 'xue']

    statistic_play_info = read_json(STATISTIC_PLAY_COUNT_FILE)
    good_video_list = statistic_play_info.get('good_video_list', [])
    good_video_list.sort(key=lambda x: len(x.get("choose_reason", [])), reverse=True)
    user_config = read_json(r'W:\project\python_project\auto_video\config\user_config.json')
    user_type_info = user_config.get('user_type_info', {})
    for video_type, user_list in user_type_info.items():
        # 删除没有在need_process_users列表中的用户
        user_list[:] = [user for user in user_list if user in need_process_users]

    final_video_list = []
    user_statistic_info = read_json(USER_STATISTIC_INFO_PATH)
    user_count_info = defaultdict(dict)
    total_need_count = 0
    for user_name in need_process_users:
        total_count = user_statistic_info.get(user_name, {}).get('today_process', 0)
        platform_upload_count = user_statistic_info.get(user_name, {}).get('platform_upload_count', 0)
        if user_name in simple_need_process_users:
            user_count_info[user_name]['need_count'] = max(1 - total_count - platform_upload_count, 0)
            total_need_count += max(1 - total_count - platform_upload_count, 0)
            user_count_info[user_name]['send_count'] = 0
            print(f"用户 {user_name} 为简化用户，今日需要处理 {total_need_count} 个任务。")
            continue
        total_count = user_statistic_info.get(user_name, {}).get('today_process', 0)
        target_count = 1
        need_count = min(max(target_count - total_count, 0), 0)
        user_count_info[user_name]['need_count'] = need_count
        total_need_count += need_count
        user_count_info[user_name]['send_count'] = 0
        print(f"用户 {user_name} 今日已收到 {total_count} 个任务，还需处理 {need_count} 个。才能够达到目标 {target_count} 个")

    user_type_count_info = {}
    for video_info in good_video_list:
        user_name = video_info.get('userName', 'dahao')
        user_type = get_user_type(user_name)
        if user_type not in user_type_count_info:
            user_type_count_info[user_type] = 0
        user_type_count_info[user_type] += 1
        user_list = user_type_info.get(user_type, [])
        single_count = 1
        same_count = video_info.get('same_count', 1)
        if same_count > 1:
            single_count += 1
        final_score = video_info.get('final_score', 0)
        if final_score > 5000:
            single_count += 1

        # 最多随机选择3个
        final_user_list = random.sample(user_list, min(len(user_list), single_count))

        for user_name in final_user_list:
            # 深拷贝一份video_info
            copy_video_info = video_info.copy()
            copy_video_info['target_user_name'] = user_name
            final_video_list.append(copy_video_info)


    success_count = 0
    print(user_count_info)
    for video_info in final_video_list:
        final_score = video_info.get('final_score', 0)
        target_user_name = video_info.get('target_user_name', 'dahao')
        if user_count_info[target_user_name]['send_count'] >= user_count_info[target_user_name]['need_count'] and final_score < 20000:
            print(f"用户 {target_user_name} 已达到今日发送目标，跳过后续视频。")
            continue
        creation_guidance_info = video_info.get('creation_guidance_info')
        upload_params = video_info.get('upload_params', {})
        title = upload_params.get('title', '变得有吸引力一点')
        creation_guidance_info['creative_guidance'] = f"标题尽量体现: {title}"
        original_url_info_list = video_info.get('original_url_info_list', [])
        play_load = {
        'userName': target_user_name,
            'global_settings':creation_guidance_info,
            'video_list':original_url_info_list

        }

        play_comment_info_list = video_info.get('play_comment_info_list')
        print(f"正在往 {target_user_name} 发送 {title} 数据为 {play_comment_info_list[-1]}")
        data_info = send_good_video_quest(play_load)
        if '新任务已成功创建' in str(data_info):
            user_count_info[target_user_name]['send_count'] += 1
            success_count += 1

    # 梳理出 send_count 大于0的用户
    fail_user_count_info = {k: v for k, v in user_count_info.items() if v['send_count'] <= 0}
    print(f"总共收集了 {len(final_video_list)} 个优质视频。成功发送了 {success_count} 个视频。 总共需要 {total_need_count} 个视频 {user_type_count_info} 当前时间 {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(fail_user_count_info)
    print(user_count_info)







if __name__ == "__main__":
    if __name__ == "__main__":
        while True:
            try:
                send_good_video()
            except Exception as e:
                traceback.print_exc()
                print(f"出错了: {e}")

            # 暂停 30 分钟 (30 * 60 秒)
            print("等待30分钟后再次运行...")
            time.sleep(60 * 30)

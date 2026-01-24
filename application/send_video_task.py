import random
import time
import traceback
from collections import defaultdict
from datetime import datetime, timedelta

import requests
import json

from application.dig_video import query_all_material_videos
from application.video_common_config import USER_STATISTIC_INFO_PATH, STATISTIC_PLAY_COUNT_FILE, \
    DIG_HOT_VIDEO_PLAN_FILE, VIDEO_MAX_RETRY_TIMES, TaskStatus
from utils.common_utils import read_json, save_json, get_user_type, gen_true_type_and_tags, \
    has_continuous_common_substring, has_long_common_substring
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
    url = "http://127.0.0.1:5001/one-click-generate"

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

def send_good_video(manager):
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
    need_process_users = ['lin', 'dahao', 'zhong', "qizhu", 'mama', 'xiaosu', 'jie', 'qiqixiao', 'yang', 'xue', 'danzhu', 'ruruxiao', 'yuhua', 'junyuan', 'xiaoxiaosu', 'junda']
    simple_need_process_users = ['yang', 'xue', 'danzhu', 'ruruxiao', 'yuhua', 'junyuan', 'xiaoxiaosu']

    statistic_play_info = read_json(STATISTIC_PLAY_COUNT_FILE)
    good_video_list = statistic_play_info.get('good_video_list', [])
    good_video_list.sort(key=lambda x: len(x.get("choose_reason", [])), reverse=True)
    good_video_list = sorted(good_video_list, key=lambda x: (x.get('final_score', 0)), reverse=True)

    user_config = read_json(r'W:\project\python_project\auto_video\config\user_config.json')
    user_type_info = user_config.get('user_type_info', {})
    user_tags_info = user_config.get('user_tags', {})
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
        target_count = 5
        if (5 <= datetime.now().hour < 24):
            target_count = 2
        need_count = max(target_count - total_count, 0)
        user_count_info[user_name]['need_count'] = need_count
        total_need_count += need_count
        user_count_info[user_name]['send_count'] = 0
        print(f"用户 {user_name} 今日已收到 {total_count} 个任务，还需处理 {need_count} 个。才能够达到目标 {target_count} 个")

    need_user_count_info = {k: v for k, v in user_count_info.items() if v['need_count'] > 0}
    need_user_list = need_user_count_info.keys()
    user_type_count_info = {}
    for video_info in good_video_list:

        upload_info_list = video_info.get('upload_info', [])
        video_type, tags_info = gen_true_type_and_tags(upload_info_list)
        tags = tags_info.keys()
        user_name = video_info.get('userName', 'dahao')
        user_type = get_user_type(user_name)
        if user_type not in user_type_count_info:
            user_type_count_info[user_type] = 0
        user_type_count_info[user_type] += 1
        user_list = user_type_info.get(user_type, [])
        single_count = 3
        same_count = video_info.get('same_count', 1)
        if same_count > 1:
            single_count += 3
        final_score = video_info.get('final_score', 0)
        if final_score > 1000:
            single_count += 3
        create_time = video_info.get('created')
        pre_timestamp = int(time.time()) - 3 * 3600
        if create_time > pre_timestamp:
            single_count += 3
        video_id_list = video_info.get('video_id_list', [])
        query_2 = {
            'video_id_list': video_id_list
        }
        exist_tasks = manager.find_by_custom_query(manager.tasks_collection, query_2)
        upload_params = video_info.get('upload_params', {})
        title = upload_params.get('title', '变得有吸引力一点')
        if "太太当腻了？章泽天新播" in title:
            print()
        create_time_list = [task_info.get('created', 0) for task_info in exist_tasks]
        max_timestamp = max(create_time_list)
        can_use_count = max(single_count - len(exist_tasks), 0)

        if (int(time.time()) - max_timestamp) > 24 * 3600:
            can_use_count = max(single_count - len(exist_tasks), 1)
            print(f"{title} 超过24小时，允许再创建一个任务 最近的时间为 {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(max_timestamp))} ")
        can_use_count = min(2, can_use_count)

        print(f"{title} final_score:{final_score} {user_type} 总共能够发送 {single_count} 个任务，当前已有 {len(exist_tasks)}，最终还能够创建 {can_use_count} 个任务")
        set_user_list = set(user_list)
        set_need_user_list = set(need_user_list)
        common_user_list = list(set_user_list & set_need_user_list)
        filter_user_list = []
        for user in common_user_list:
            user_tags = user_tags_info.get(user, [])
            if not user_tags:
                filter_user_list.append(user)
                continue
            found_any, results = has_continuous_common_substring(user_tags, tags, 1)
            if found_any:
                # print(results)
                filter_user_list.append(user)

        # 最多随机选择3个
        final_user_list = random.sample(filter_user_list, min(len(filter_user_list), can_use_count))

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
        if user_count_info[target_user_name]['send_count'] >= user_count_info[target_user_name]['need_count'] and final_score < 5000:
            # print(f"用户 {target_user_name} 已达到今日发送目标，跳过后续视频。")
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
        print(f"正在往 {target_user_name} 发送 {title}  数据为 {play_comment_info_list[-1]} final_score：{final_score} same_count: {video_info.get("same_count")}   video_id_list: {video_info.get('video_id_list')} 当前时间 {time.strftime('%Y-%m-%d %H:%M:%S')}")
        data_info = send_good_video_quest(play_load)
        if '新任务已成功创建' in str(data_info):
            user_count_info[target_user_name]['send_count'] += 1
            success_count += 1

    # 梳理出 send_count 大于0的用户
    fail_user_count_info = {k: v for k, v in user_count_info.items() if v['send_count'] <= 0}
    success_user_count_info = {k: v for k, v in user_count_info.items() if v['send_count'] > 0}
    need_user_count_info = {k: v for k, v in user_count_info.items() if v['need_count'] > 0}
    other_user_count_info = {k: v for k, v in user_count_info.items() if v['need_count'] != v['send_count']}

    print(f"总共收集了 {len(final_video_list)} 个优质视频。成功发送了 {success_count} 个视频。 总共需要 {total_need_count} 个视频 {user_type_count_info} 当前时间 {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"需要的用户详情{need_user_count_info}")
    print(f"没完成目标用户详情{other_user_count_info}")
    print(f"有成功的用户详情{success_user_count_info}")
    print(f"没发送视频用户详情{fail_user_count_info}")
    print(f"完整的用户详情{user_count_info}")


def get_need_count(need_process_users):
    user_count_info = defaultdict(dict)
    target_count = 18
    user_statistic_info = read_json(USER_STATISTIC_INFO_PATH)

    for user_name in need_process_users:
        platform_upload_count = user_statistic_info.get(user_name, {}).get('platform_upload_count', 0)
        total_count = user_statistic_info.get(user_name, {}).get('today_process', 0)
        need_count = max(target_count - total_count - platform_upload_count, 0)
        user_count_info[user_name]['need_count'] = need_count
        user_count_info[user_name]['send_count'] = 0
        print(
            f"用户 {user_name} 今日已收到 {total_count} 个任务，还需处理 {need_count} 个。才能够达到目标 {target_count} 个")

    return user_count_info
def send_dig_video(manager):
    """
    进行挖掘的视频投递
    :return:
    """
    need_process_users = ['junda', 'lin', 'dahao', 'zhong', "qizhu", 'mama', 'xiaosu', 'jie', 'qiqixiao']

    user_count_info = get_need_count(need_process_users)
    need_user_count_info = {k: v for k, v in user_count_info.items() if v['need_count'] > 0}
    need_user_list = need_user_count_info.keys()

    exist_video_plan_info = read_json(DIG_HOT_VIDEO_PLAN_FILE)
    all_plan_info_list = []
    for hot_key, plan_info_list in exist_video_plan_info.items():
        # 只保留score 90分以上的plan_info_list
        filtered_plan_info_list = [plan_info for plan_info in plan_info_list if plan_info.get('score', 0) >= 90]
        for filtered_plan_info in filtered_plan_info_list:
            filtered_plan_info['hot_key'] = hot_key
        all_plan_info_list.extend(filtered_plan_info_list)
    # 再按照final_score降序排序
    all_plan_info_list = sorted(all_plan_info_list, key=lambda x: x.get('final_score', 0), reverse=True)
    user_config = read_json(r'W:\project\python_project\auto_video\config\user_config.json')
    user_type_info = user_config.get('user_type_info', {})

    select_plan_infp_list = []
    for plan_info in all_plan_info_list:
        user_type = plan_info.get('user_type_info', '99')
        video_id_list = plan_info.get('video_id_list', [])
        query_2 = {
            'video_id_list': video_id_list
        }
        exist_tasks = manager.find_by_custom_query(manager.tasks_collection, query_2)
        can_use_count = max(2 - len(exist_tasks), 0)
        final_score = plan_info.get('final_score', 0)

        video_theme = plan_info.get('video_theme')

        print(f"{video_theme} video_id_list: {video_id_list} final_score {final_score} 已存在 {len(exist_tasks)}个任务 当前还能够创建{can_use_count} 个任务")
        user_list = user_type_info.get(user_type, [])
        set_user_list = set(user_list)
        set_need_user_list = set(need_user_list)
        common_user_list = list(set_user_list & set_need_user_list)
        final_user_list = random.sample(common_user_list, min(len(common_user_list), can_use_count))
        for user_name in final_user_list:
            copy_plan_info = plan_info.copy()
            copy_plan_info['target_user_name'] = user_name
            select_plan_infp_list.append(copy_plan_info)

    success_count = 0
    for plan_info in select_plan_infp_list:
        hot_key = plan_info.get('hot_key', '')
        video_id_list = plan_info.get('video_id_list', [])
        final_score = plan_info.get('final_score', 0)
        target_user_name = plan_info.get('target_user_name', 'dahao')

        if 'hot_key_list' not in user_count_info[target_user_name]:
            user_count_info[target_user_name]['hot_key_list'] = []

        if hot_key in user_count_info[target_user_name]['hot_key_list']:
            # print(f"用户 {target_user_name} 已经发送过 热门关键词 {hot_key} ，跳过。")
            continue
        user_count_info[target_user_name]['hot_key_list'].append(hot_key)


        if user_count_info[target_user_name]['send_count'] >= user_count_info[target_user_name]['need_count'] and final_score < 5000:
            # print(f"用户 {target_user_name} 已达到今日发送目标，跳过后续视频。")
            continue

        video_theme = plan_info.get('video_theme')
        story_outline = plan_info.get('story_outline')
        creative_guidance = f"视频主题: {video_theme}，剧情大纲: {story_outline}"
        creation_guidance_info = {
            "video_type": "通用",
            "retention_ratio": "free",
            "is_need_audio_replace": True,
            "is_need_scene_title": True,
            "is_need_commentary": True,
            "schedule_date": "2026-01-05",
            "creative_guidance": creative_guidance,
            "is_origin": True
        }
        video_list = []
        video_info_map = {}
        video_info_list = manager.find_materials_by_ids(video_id_list)
        for video_info in video_info_list:
            video_id = video_info.get('video_id')
            video_info_map[video_id] = video_info
        for video_id in video_id_list:
            video_info = video_info_map.get(video_id, {})
            extra_info = video_info.get('extra_info', {})
            video_list.append(extra_info)

        play_load = {
        'userName': target_user_name,
            'global_settings': creation_guidance_info,
            'video_list': video_list

        }
        print(f"正在往 {target_user_name} 发送 挖掘视频 final_score：{final_score} creative_guidance{creative_guidance}   video_id_list: {video_id_list} 当前时间 {time.strftime('%Y-%m-%d %H:%M:%S')}")
        data_info = send_good_video_quest(play_load)
        if '新任务已成功创建' in str(data_info):
            user_count_info[target_user_name]['send_count'] += 1
            success_count += 1

    # 计算total_send_count
    total_need_count = sum(
        user_info.get('need_count', 0)
        for user_info in user_count_info.values()
    )

    print(f"总共收集了 {len(select_plan_infp_list)} 个优质视频。成功发送了 {success_count} 个视频。 总共需要 {total_need_count} 个视频  当前时间 {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"需要的用户详情{need_user_count_info}")
    other_user_count_info = {k: v for k, v in user_count_info.items() if v['need_count'] != v['send_count']}

    print(f"没完成目标用户详情{other_user_count_info}")
    success_user_count_info = {k: v for k, v in user_count_info.items() if v['send_count'] > 0}

    print(f"有成功的用户详情{success_user_count_info}")
    fail_user_count_info = {k: v for k, v in user_count_info.items() if v['send_count'] <= 0}

    print(f"没发送视频用户详情{fail_user_count_info}")
    print(f"完整的用户详情{user_count_info}")


def gen_standard_video_info_by_statistic_data(good_video_list):
    """
    将统计数据变成标准的待投稿数据
    :return:
    """
    standard_video_list = []

    for video_info in good_video_list:
        upload_params = video_info.get('upload_params', {})
        hot_video = upload_params.get('title', '变得有吸引力一点')
        video_id_list = video_info.get('video_id_list', [])
        video_theme = hot_video
        user_name = video_info.get('userName', '')
        video_type = get_user_type(user_name)
        final_score = video_info.get('final_score', 0)
        dig_type = 'exist_video'
        creative_guidance = f"标题尽量体现: {hot_video}"

        temp_dict = {
            'video_id_list': video_id_list,
            'video_theme': video_theme,
            'hot_topic': video_theme,
            'video_type': video_type,
            'final_score': final_score,
            'dig_type': dig_type,
            "creative_guidance": creative_guidance,
        }
        temp_dict.update(video_info)
        standard_video_list.append(temp_dict)
    return standard_video_list

def gen_standard_video_info_by_dig_data(plan_info):
    """
    将挖掘的数据变成标准化的待投稿数据
    :param plan_info_list:
    :return:
    """
    standard_video_list = []
    for hot_topic, plan_info_list in plan_info.items():
        for plan_info in plan_info_list:
            score = plan_info.get('score', 0)
            if score < 95:
                continue
            video_id_list = plan_info.get('video_id_list', [])
            video_theme = plan_info.get('video_theme', '')
            video_type = plan_info.get('video_type', '')
            final_score = plan_info.get('final_score', 0)
            dig_type = plan_info.get('dig_type', 0)
            creative_guidance = plan_info.get('creative_guidance', '')
            now_timestamp = int(datetime.now().timestamp())

            dig_time = plan_info.get('dig_time', now_timestamp - 3600 * 5)
            update_time = plan_info.get('update_time', 0)
            # 获取当前的timestamp
            # 计算相差的小时数量
            hours_diff = (now_timestamp - dig_time) / 3600
            final_score = (100 - hours_diff) / 100 * final_score
            if final_score < 10:
                continue

            temp_dict = {
                'video_id_list': video_id_list,
                'video_theme': video_theme,
                'hot_topic': hot_topic,
                'video_type': video_type,
                'final_score': final_score,
                'dig_type': dig_type,
                'dig_time': dig_time,
                'update_time': update_time,
                "creative_guidance": creative_guidance,
            }
            standard_video_list.append(temp_dict)
    return standard_video_list


def build_need_upload_video():
    """
    生成统一的规范的待投稿数据
    :return:
    """
    statistic_play_info = read_json(STATISTIC_PLAY_COUNT_FILE)
    good_video_list = statistic_play_info.get('good_video_list', [])
    standard_good_video_list = gen_standard_video_info_by_statistic_data(good_video_list)

    exist_video_plan_info = read_json(DIG_HOT_VIDEO_PLAN_FILE)
    standard_dig_video_list = gen_standard_video_info_by_dig_data(exist_video_plan_info)
    combined_video_list = standard_good_video_list + standard_dig_video_list
    print(f"总共收集了 {len(standard_good_video_list)} 个来源于统计视频和 {len(standard_dig_video_list)} 个来源于挖掘视频，合计 {len(combined_video_list)} 个待投稿视频。")

    # 将 combined_video_list 按照 final_score 降序排序
    combined_video_list = sorted(combined_video_list, key=lambda x: x.get('final_score', 0), reverse=True)
    return combined_video_list


def gen_user_detail_upload_info(manager, user_list):
    """
    获取用户今日详细的投稿数据，确保每个用户都有完整的四个字段
    """

    # 1. 初始化结果字典，确保 user_list 中每个用户都有基础结构
    user_detail_upload_info = {
        user_name: {
            "dig_type": {},
            "hot_topic": {},
            "success_count": 0,
            "total_count": 0,
            "unique_video_id_list":[]
        } for user_name in user_list
    }

    now = datetime.now()
    today_midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)

    query_2 = {
        "userName": {"$in": user_list},
        "failed_count": {"$lt": VIDEO_MAX_RETRY_TIMES + 1},
        "create_time": {"$gt": today_midnight}
    }
    task_list = manager.find_by_custom_query(manager.tasks_collection, query_2)

    # 2. 填充数据
    for task_info in task_list:
        user_name = task_info.get('userName')

        # 容错：如果数据库查出了不在 user_list 里的用户（虽然 query 限制了，但为了健壮性可保留）
        if user_name not in user_detail_upload_info:
            continue

        user_data = user_detail_upload_info[user_name]
        creation_guidance_info = task_info.get('creation_guidance_info', {})

        # 统计 dig_type
        dig_type = creation_guidance_info.get('dig_type', 'unknown')
        user_data["dig_type"][dig_type] = user_data["dig_type"].get(dig_type, 0) + 1

        # 统计 hot_topic
        hot_topic = creation_guidance_info.get('hot_topic', '未知主题')
        user_data["hot_topic"][hot_topic] = user_data["hot_topic"].get(hot_topic, 0) + 1

        video_id_list = task_info.get('video_id_list', [])
        for video_id in video_id_list:
            if video_id not in user_data["unique_video_id_list"]:
                user_data["unique_video_id_list"].append(video_id)

        # 统计状态
        status = task_info.get('status')
        if status == TaskStatus.UPLOADED:
            user_data["success_count"] += 1

        user_data["total_count"] += 1


    user_statistic_info = read_json(USER_STATISTIC_INFO_PATH)
    user_config = read_json(r'W:\project\python_project\auto_video\config\user_config.json')

    simple_need_process_users = user_config.get('simple_need_process_users', [])
    max_exist_similar_count, max_total_count = get_send_count_by_hour()

    for user_name, detail_info in user_detail_upload_info.items():
        if user_name == "xiaoxiaosu":
            print()
        total_count = user_statistic_info.get(user_name, {}).get('today_process', 0)
        platform_upload_count = user_statistic_info.get(user_name, {}).get('platform_upload_count', 0)
        total_today = total_count + platform_upload_count
        target_count = max_total_count
        if user_name in simple_need_process_users:
            target_count = 1
        need_count = max(target_count - total_today, 0)
        need_count = min(need_count, 1)
        detail_info['need_count'] = need_count
        detail_info['send_count'] = 0




    return user_detail_upload_info

def get_available_count(manager, video_info):
    """
    获取此组合还可以发的数量
    :param manager:
    :param video_info:
    :return:
    """
    single_count = 1
    final_score = video_info.get('final_score', 0)
    if final_score > 1000:
        single_count += 1
    if final_score > 3000:
        single_count += 2
    if final_score > 5000:
        single_count += 3
    if final_score > 10000:
        single_count += 5
    video_id_list = video_info.get('video_id_list', [])
    query_2 = {
        'video_id_list': video_id_list,
        "failed_count": {"$lt": VIDEO_MAX_RETRY_TIMES + 1},
    }
    exist_tasks = manager.find_by_custom_query(manager.tasks_collection, query_2)
    # 1. 获取当前时间
    now = datetime.now()
    create_time_list = [task_info.get('create_time') for task_info in exist_tasks if task_info.get('create_time')]
    if create_time_list:
        latest_create_time = max(create_time_list)
    else:
        latest_create_time = now
    is_within_3_hours = latest_create_time >= (now - timedelta(hours=3))
    if is_within_3_hours and final_score > 1000:
        single_count += 1
    if len(exist_tasks) > 1:
        single_count += 1

    can_use_count = max(single_count - len(exist_tasks), 0)
    is_high_score = final_score > 5000
    is_within_12_hours = latest_create_time >= (now - timedelta(hours=12))

    if is_high_score and not is_within_12_hours:
        can_use_count = max(can_use_count, 1)
        print(f"高分视频{ final_score}，{video_info.get('hot_topic')} 允许再创建一个任务 最近的时间为 {latest_create_time.strftime('%Y-%m-%d %H:%M:%S')} ")

    video_info['exist_count'] = len(exist_tasks)
    video_info['single_count'] = single_count
    video_info['can_use_count'] = can_use_count

    video_info['latest_create_time'] = latest_create_time.strftime('%Y-%m-%d %H:%M:%S')
    # if "资本反噬实录" in video_info.get('hot_topic', ''):
    #     print()




    return can_use_count

def check_user_tags(user_name, video_info, all_video_info):
    """
    检查用户的指定tag列表是否符合要求
    :param user_name:
    :param video_info:
    :return:
    """
    user_config = read_json(r'W:\project\python_project\auto_video\config\user_config.json')
    user_tags_info = user_config.get('user_tags', {})
    user_tags = user_tags_info.get(user_name, [])
    if not user_tags:
        return True

    user_tags_info = {}

    for tag in user_tags:
        user_tags_info[tag] = 1

    all_video_tags = {}

    video_id_list = video_info.get('video_id_list', [])
    for video_id in video_id_list:
        video_detail_info = all_video_info.get(video_id, {})
        video_tags = video_detail_info.get('tags_info', {})
        for tag, count in video_tags.items():
            if tag not in all_video_tags:
                all_video_tags[tag] = 0
            all_video_tags[tag] += count

    total_score = 0
    common_str_list = []
    for v_tag, v_weight in all_video_tags.items():
        for t_tag, t_weight in user_tags_info.items():
            # 调用外部匹配函数
            has_comm, common_str = has_long_common_substring(v_tag, t_tag)
            if has_comm:
                # 双方都有权重，乘积累加
                total_score += v_weight * t_weight
                common_str_list.append((v_tag, t_tag))
    if total_score > 2:
        return True
    return False


def get_send_count_by_hour():
    """
    通过时间范围来确定几个比较重要的变量信息
    :return:
    """
    current_hour = datetime.now().hour
    if 0 <= current_hour < 5:
        max_exist_similar_count = 2
        max_total_count = 10
    elif 5 <= current_hour < 12:
        max_exist_similar_count = 7
        max_total_count = 15
    elif 12 <= current_hour < 18:
        max_exist_similar_count = 9
        max_total_count = 17
    elif 18 <= current_hour < 22:
        max_exist_similar_count = 11
        max_total_count = 19
    else:
        max_exist_similar_count = 13
        max_total_count = 21

    return max_exist_similar_count, max_total_count


def match_user(user_detail_upload_info, video_info, all_video_info):
    """
    获取匹配的用户列表,主要是进行主题太多的检查以及相似稿件的数量
    :param user_detail_upload_info:
    :param video_info:
    :return: matched_user (list), detail_match_info (dict)
    """
    matched_user = []
    # 新增：用于存放不匹配的原因 {user_name: reason}
    detail_match_info = {}

    video_type = video_info.get('video_type', '')
    final_score = video_info.get('final_score', 0)
    hot_topic = video_info.get('hot_topic', '')
    dig_type = video_info.get('dig_type', 'exist_video')
    high_similar_dig_type_list = ['exist_video', 'exist_video_dig']
    hot_topic_count = 1
    # 假设 get_send_count_by_hour() 定义在外部或已导入
    max_exist_similar_count, max_total_count = get_send_count_by_hour()
    user_config = read_json(r'W:\project\python_project\auto_video\config\user_config.json')

    simple_need_process_users = user_config.get('simple_need_process_users', [])


    is_high_score = final_score > 5000
    if is_high_score:
        hot_topic_count += 1

    for user_name, detail_info in user_detail_upload_info.items():
        need_count = detail_info.get('need_count', 0)
        if user_name in simple_need_process_users and need_count < 1:
            detail_match_info[user_name] = "简化用户且需求量不足"
            continue
        # 1. 检查需求量
        if need_count <= 0 and not is_high_score:
            detail_match_info[user_name] = "需求量不足且非高分稿件"
            continue

        # 2. 检查用户类型
        user_type = get_user_type(user_name)  # 假设函数已定义
        if user_type != video_type:
            # detail_match_info[user_name] = f"用户类型不匹配: 用户({user_type}) vs 视频({video_type})"
            continue

        # 3. 检查用户标签
        if check_user_tags(user_name, video_info, all_video_info) is False:  # 假设函数已定义
            detail_match_info[user_name] = "用户标签校验未通过"
            continue

        video_id_list = video_info.get('video_id_list', [])
        exist_video_id_list = detail_info.get('unique_video_id_list', [])

        # 4. 检查视频ID是否已存在
        all_exist = True
        for video_id in video_id_list:
            if video_id not in exist_video_id_list:
                all_exist = False
                break
        if all_exist:
            detail_match_info[user_name] = "视频ID已存在于该用户列表"
            continue

        # 5. 检查是否重复题材超过限制
        hot_topic_info = detail_info.get('hot_topic', {})
        exist_hot_topic_count = hot_topic_info.get(hot_topic, 0)
        if exist_hot_topic_count >= hot_topic_count:
            detail_match_info[user_name] = f"热门题材({hot_topic})数量超限: {exist_hot_topic_count}/{hot_topic_count}"
            continue

        # 6. 检查来源是否超过限制
        if dig_type in high_similar_dig_type_list:
            exist_similar_count = 0
            for similar_dig_type in high_similar_dig_type_list:
                exist_similar_count += detail_info.get('dig_type', {}).get(similar_dig_type, 0)
            if exist_similar_count >= max_exist_similar_count:
                detail_match_info[user_name] = f"相似来源稿件超限: {exist_similar_count}/{max_exist_similar_count}"
                continue

        # 通过所有检查
        matched_user.append(user_name)

    return matched_user, detail_match_info



def get_proper_user_list(manager, user_detail_upload_info, video_info, used_video_list, all_video_info):
    """
    直接获取时候投稿video_info的用户列表
    :param user_detail_upload_info:
    :param video_info:
    :return:
    """
    video_id_list = video_info.get('video_id_list', [])
    is_all_used = True
    for video_id in video_id_list:
        if video_id not in used_video_list:
            is_all_used = False
        used_video_list.append(video_id)
    used_video_list = list(set(used_video_list))
    # if is_all_used:
    #     video_info['reason'] = "视频全部使用过，跳过"
    #     return []

    can_use_count = get_available_count(manager, video_info)
    match_user_list, detail_match_info = match_user(user_detail_upload_info, video_info, all_video_info)
    video_info['match_user_list'] = match_user_list
    video_info['detail_match_info'] = detail_match_info

    sample_size = min(len(match_user_list), can_use_count)
    sample_size = min(sample_size, 1)

    # 3. 随机选择不重复的元素
    final_list = random.sample(match_user_list, sample_size) if sample_size > 0 else []
    return final_list


def upload_video(manager, video_info, user_detail_upload_info):
    """
    进行投稿
    :param manager:
    :param video_info:
    :return:
    """
    final_score = video_info.get('final_score', 0)
    is_high_score = final_score > 5000
    dig_type = video_info.get('dig_type', 'exist_video')
    user_name = video_info.get('user_name', 'dahao')
    hot_topic = video_info.get('hot_topic', '')
    high_similar_dig_type_list = ['exist_video', 'exist_video_dig']
    video_id_list = video_info.get('video_id_list', [])
    schedule_date = "2026-01-05"
    if dig_type in high_similar_dig_type_list:
        schedule_date = "2026-01-01"
    detail_upload_info = user_detail_upload_info.get(user_name, {})

    send_count = detail_upload_info.get('send_count')
    need_count = detail_upload_info.get('need_count')
    if send_count < need_count or is_high_score:
        if send_count >= need_count + 1:
            return False

        creative_guidance = video_info.get('creative_guidance', '')
        creation_guidance_info = {
            "video_type": "通用",
            "retention_ratio": "free",
            "is_need_audio_replace": True,
            "is_need_scene_title": True,
            "is_need_commentary": True,
            "schedule_date": schedule_date,
            "creative_guidance": creative_guidance,
            "dig_type": dig_type,
            "hot_topic": hot_topic,
        }
        video_list = []
        video_info_map = {}
        video_info_list = manager.find_materials_by_ids(video_id_list)
        for video_info in video_info_list:
            video_id = video_info.get('video_id')
            video_info_map[video_id] = video_info
        for video_id in video_id_list:
            video_info = video_info_map.get(video_id, {})
            extra_info = video_info.get('extra_info', {})
            video_list.append(extra_info)

        play_load = {
        'userName': user_name,
            'global_settings': creation_guidance_info,
            'video_list': video_list

        }
        print(f"正在往 {user_name} 发送 {dig_type}类型视频 final_score：{final_score}   video_id_list: {video_id_list} hot_topic：{hot_topic} creative_guidance{creative_guidance}当前时间 {time.strftime('%Y-%m-%d %H:%M:%S')}")
        data_info = send_good_video_quest(play_load)
        if '新任务已成功创建' in str(data_info):
            detail_upload_info['send_count'] += 1
            return True
    return False


def print_statistics(data):
    """
    统计并打印字典中的 need_count 和 send_count 信息
    """
    total_need_sum = 0
    total_send_sum = 0

    # 用于存储未达标的 key 信息
    insufficient_items = []

    for key, info in data.items():
        # 获取当前项的数量，如果不存在则默认为0
        need = info.get('need_count', 0)
        send = info.get('send_count', 0)

        # 累加总数
        total_need_sum += need
        total_send_sum += send

        # 判断实际数量是否小于需要数量
        if send < need:
            insufficient_items.append({
                'key': key,
                'need': need,
                'send': send
            })

    # --- 打印结果 ---

    # 1. 总共需要的数量，实际发送的数量
    print("=== 统计概览 ===")
    print(f"总共需要的数量 (Total Need): {total_need_sum}")
    print(f"实际发送的数量 (Total Send): {total_send_sum}")
    print("-" * 40)

    # 2. 实际数量小于需要数量的 key 及具体详情
    print("=== 未达标项目详情 (Send < Need) ===")
    if not insufficient_items:
        print("所有项目均已达标！")
    else:
        # 为了美观，使用格式化对齐打印
        header = f"{'Key':<15} | {'Need Count':<12} | {'Send Count':<12}"
        print(header)
        print("-" * len(header))

        for item in insufficient_items:
            print(f"{item['key']:<15} | {item['need']:<12} | {item['send']:<12}")

def send_good_plan(manager):
    """
    进行投稿
    :param manager:
    :return:
    """
    need_process_users = ['hong', 'lin', 'dahao', 'zhong', "qizhu", 'mama', 'yang', 'xue', 'danzhu', 'ruruxiao', 'yuhua', 'junyuan', 'xiaoxiaosu', 'junda']
    user_detail_upload_info = gen_user_detail_upload_info(manager, need_process_users)
    all_video_info = query_all_material_videos(manager, False)



    # 获取需要投稿的数据
    to_upload_video_list = build_need_upload_video()
    used_video_list = []
    final_video_list = []

    for video_info in to_upload_video_list:
        chosen_user_list = get_proper_user_list(manager, user_detail_upload_info, video_info, used_video_list, all_video_info)
        for user_name in chosen_user_list:
            copy_video_info = video_info.copy()
            copy_video_info['user_name'] = user_name
            final_video_list.append(copy_video_info)

    print(f"总共选择了 {len(final_video_list)} 个视频用于投稿。")
    success_count = 0
    # 进行最终的投稿
    # 先按照 final_score 降序排序
    final_video_list = sorted(final_video_list, key=lambda x: x.get('final_score', 0), reverse=True)
    for video_info in final_video_list:
        send_flag = upload_video(manager, video_info, user_detail_upload_info)
        if send_flag:
            success_count += 1
    print_statistics(user_detail_upload_info)







if __name__ == "__main__":
    mongo_base_instance = gen_db_object()
    manager = MongoManager(mongo_base_instance)
    while True:
        try:
            send_good_plan(manager)
        except Exception as e:
            traceback.print_exc()
            print(f"出错了: {e}")

        # 暂停 30 分钟 (30 * 60 秒)
        print(f"等待30分钟后再次运行...当前时间 {time.strftime('%Y-%m-%d %H:%M:%S')}")
        time.sleep(60 * 30)

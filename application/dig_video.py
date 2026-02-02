# -- coding: utf-8 --
""":authors:
    zhuxiaohu
:create_date:
    2026/1/19 23:59
:last_date:
    2026/1/19 23:59
:description:
    进行素材库的视频挖掘，挖掘可能不错的视频方案
"""
import json
import os
import random
import time
import traceback

from application.video_common_config import ALL_MATERIAL_VIDEO_INFO_PATH, BLOCK_VIDEO_BVID_FILE, get_tags_info, \
    DIG_HOT_VIDEO_PLAN_FILE, STATISTIC_PLAY_COUNT_FILE, is_contain_owner_speaker, analyze_scene_content, \
    BLOCK_VIDEO_ID_FILE, DIG_HOT_VIDEO_PLAN_ARCHIVE_FILE
from utils.common_utils import read_json, save_json, has_long_common_substring, read_file_to_str, string_to_object, \
    gen_true_type_and_tags, get_user_type
from utils.gemini_cli import ask_gemini
from utils.mongo_base import gen_db_object
from utils.mongo_manager import MongoManager

NEED_REFRESH = False


def query_all_material_videos(manager, is_need_refresh):
    """
    查询所有的素材视频，已剔除黑名单素材
    :return:
    """
    import time  # 确保引入time模块
    start_time = time.time()  # [新增] 记录开始时间

    global NEED_REFRESH
    local_material_video_info = read_json(ALL_MATERIAL_VIDEO_INFO_PATH)

    # 判断ALL_MATERIAL_VIDEO_INFO_PATH这个文件上次修改时间是否超过1天，超过1天则重新从数据库中查询
    if os.path.exists(ALL_MATERIAL_VIDEO_INFO_PATH):
        modify_time = os.path.getmtime(ALL_MATERIAL_VIDEO_INFO_PATH)
        # 86400秒 = 1天
        if time.time() - modify_time < 86400 and not NEED_REFRESH and not is_need_refresh:
            print(
                f"all_need_plan_video_info.json 缓存文件存在且在一天之内，直接读取。当前数据量: {len(local_material_video_info)}, 耗时: {time.time() - start_time:.4f}s")  # [修改] 增加耗时和数据量打印
            return local_material_video_info


    query = {}
    local_material_video_info = {}

    all_material_list = manager.find_by_custom_query(manager.materials_collection, query)
    raw_db_count = len(all_material_list)  # [新增] 记录数据库原始查询数量

    all_task_list = manager.find_by_custom_query(manager.tasks_collection, {})

    all_video_type_map = {}
    for task_info in all_task_list:
        task_user_name = task_info.get('userName', '')
        user_type = get_user_type(task_user_name)
        video_id_list = task_info.get('video_id_list', [])
        for video_id in video_id_list:
            all_video_type_map[video_id] = user_type


    for video_info in all_material_list:
        video_id = video_info['video_id']
        logical_scene_info = video_info.get('logical_scene_info', {})
        owner_asr_info = video_info.get('owner_asr_info', [])
        max_scenes = video_info.get('extra_info', {}).get('max_scenes', 0)
        tags_info = get_tags_info(logical_scene_info)
        if tags_info:
            temp_dict = {}
            temp_dict['tags_info'] = tags_info
            temp_dict['logical_scene_info'] = logical_scene_info
            temp_dict['video_type'] = all_video_type_map.get(video_id, 'game')
            temp_dict['max_scenes'] = max_scenes
            temp_dict['owner_asr_info'] = owner_asr_info

            local_material_video_info[video_id] = temp_dict

    count_before_filter = len(local_material_video_info)  # [新增] 记录过滤前的数据总量

    # 去除黑名单素材视频
    block_video_id_list = []
    exist_block_video_info = read_json(BLOCK_VIDEO_BVID_FILE)
    all_bvid_list = list(exist_block_video_info.keys())
    query_4 = {
        "bvid": {
            "$in": all_bvid_list
        }
    }
    blocked_task_list = manager.find_by_custom_query(manager.tasks_collection, query_4)
    for task_info in blocked_task_list:
        block_video_id_list.extend(task_info.get('video_id_list', []))
    # 对block_video_id_list去重
    block_video_id_list = list(set(block_video_id_list))
    save_json(BLOCK_VIDEO_ID_FILE, block_video_id_list)  # 更新黑名单文件

    total_block_ids = len(block_video_id_list)  # [新增] 记录黑名单ID总数
    removed_count = 0  # [新增] 记录实际移除的数量

    # 删除在block_video_id_list中的视频
    for block_video_id in block_video_id_list:
        if block_video_id in local_material_video_info:
            del local_material_video_info[block_video_id]
            removed_count += 1  # [新增] 计数

    save_json(ALL_MATERIAL_VIDEO_INFO_PATH, local_material_video_info)
    NEED_REFRESH = False

    # [新增] 最终日志打印
    end_time = time.time()
    cost_time = end_time - start_time
    final_count = len(local_material_video_info)

    print("=" * 50)
    print(f"[素材查询日志] 时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    print(f"[素材查询日志] 总体耗时: {cost_time:.4f} 秒")
    print(f"[素材查询日志] 数据库原始素材数: {raw_db_count}")
    print(f"[素材查询日志] 待过滤池总数(含本地/合并后): {count_before_filter}")
    print(f"[素材查询日志] 黑名单任务数: {len(blocked_task_list)}, 涉及屏蔽视频ID数: {total_block_ids}")
    print(f"[素材查询日志] 实际命中并剔除素材数: {removed_count}")
    print(f"[素材查询日志] 最终可用素材数: {final_count}")
    print("=" * 50)

    return local_material_video_info


def process_and_sort_video_info(video_info, target_tags_info):
    """
    计算匹配得分并更新到 video_info 中，最后返回按得分降序排序的 video_info 字典。

    得分计算包含两部分：
    1. tags_info 标签加权匹配。
    2. info 整体转字符串后的全文模糊匹配（防止漏掉没有tags_info但内容匹配的数据）。

    外部依赖: has_long_common_substring(str1, str2) -> bool
    """
    for vid, info in video_info.items():
        total_score = 0
        common_str_list = []

        # --- 第一部分：基于 tags_info 的加权匹配 ---
        video_tags_info = info.get('tags_info', {})

        # 只有当 tags_info 不为空时才进行这部分计算
        if video_tags_info:
            for v_tag, v_weight in video_tags_info.items():
                for t_tag, t_weight in target_tags_info.items():
                    # 调用外部匹配函数

                    has_comm, common_str = has_long_common_substring(v_tag, t_tag)
                    if has_comm:
                        # 双方都有权重，乘积累加
                        total_score += v_weight * t_weight
                        common_str_list.append((v_tag, t_tag))
        info_str = str(info)

        for t_tag, t_weight in target_tags_info.items():
            has_comm, common_str = has_long_common_substring(info_str, t_tag)

            if has_comm:
                total_score += t_weight * t_weight
                common_str_list.append(common_str)

        # 将最终计算的总分写入字典
        info['match_score'] = total_score
        info['common_str_list'] = common_str_list

    # 按照 match_score 降序排序
    # Python 3.7+ 字典保持插入顺序，这里重构一个有序字典返回
    sorted_video_info = dict(
        sorted(video_info.items(), key=lambda x: x[1]['match_score'], reverse=True)
    )

    return sorted_video_info

def gen_target_tags(user_name='danzhu'):
    """
    生成最终目标的标签池，方便后续进行过滤素材
    :return:
    """
    user_config = read_json(r'W:\project\python_project\auto_video\config\user_config.json')
    user_tags_info = user_config.get('user_tags', {})
    user_tags_list = user_tags_info.get(user_name)

    # 将user_tags_list变成dict,key为标签，value为权重1
    target_tags = {tag: 1 for tag in user_tags_list}
    return target_tags

def build_merged_scene_list_auto(
    new_scene_info,
    owner_asr_info,
    max_scenes,
    is_need_narration=False
):
    """
    自动判断 merge_mode，然后生成 merged_scene_list
    """

    if not new_scene_info:
        return []

    if max_scenes == 1:
        merge_mode = 'global'
    else:
        if is_need_narration and is_contain_owner_speaker(owner_asr_info):
            merge_mode = 'none'
        else:
            merge_mode = 'smart'

    return analyze_scene_content(
        new_scene_info,
        owner_asr_info,
        merge_mode=merge_mode
    )


def build_prompt_data(video_info, video_type=None):
    """
    构建提示词数据
    :param video_info:
    :return:
    """
    prompt_data_info = {}
    for vid, info in video_info.items():
        scene_summary_list = []
        logical_scene_info = info.get('logical_scene_info', {})
        video_summary = logical_scene_info.get('video_summary', '')
        new_scene_info = logical_scene_info.get('new_scene_info', [])
        for new_scene in new_scene_info:
            scene_summary = new_scene.get('scene_summary', '')
            scene_summary_list.append(scene_summary)

        if video_type == 'game':
            merged_scene_list = build_merged_scene_list_auto(new_scene_info, info.get('owner_asr_info', []), info.get('max_scenes', 0), is_need_narration=True)

        prompt_entry = {
            'video_id': vid,
            'video_summary': video_summary,
            'scene_summary_list': scene_summary_list,
        }
        prompt_data_info[vid] = prompt_entry
    return prompt_data_info

def check_video_content_plan(video_content_plans, valid_video_list):
    """
    检查视频内容计划的有效性。

    Args:
        video_content_plans (list): 模型生成的计划列表
        valid_video_list (list/set/dict): 原始有效的视频ID集合，用于校验是否存在

    Returns:
        tuple: (bool, str) -> (是否通过, 错误信息)
    """

    # 0. 基础类型检查
    if not isinstance(video_content_plans, list):
        return False, "返回数据格式错误：video_content_plans 必须是一个列表"

    # 为了提高查找效率，将 valid_video_list 转换为集合 (Set)
    valid_keys_set = set(valid_video_list)

    # 定义必须存在的字段
    required_fields = {
        'video_id_list',
        'video_theme',
        'score'
    }

    for index, plan in enumerate(video_content_plans):
        # 1. 检查是否为字典
        if not isinstance(plan, dict):
            return False, f"第 {index + 1} 个方案格式错误：列表元素必须是字典"

        # 2. 检查必须包含的字段 (Missing Keys)
        # 使用 set 操作来判断 plan.keys() 是否包含所有 required_fields
        missing_keys = required_fields - set(plan.keys())
        if missing_keys:
            return False, f"第 {index + 1} 个方案缺失字段：{', '.join(missing_keys)}"

        # 3. 检查 video_keys 的有效性
        v_keys = plan.get('video_id_list')

        # 3.1 检查类型是否为列表
        if not isinstance(v_keys, list):
            return False, f"第 {index + 1} 个方案的 'video_keys' 必须是一个列表"

        # 3.2 检查长度是否大于 1 (剪辑至少需要2个视频)
        if len(v_keys) <= 1:
            return False, f"第 {index + 1} 个方案无效：'video_keys' 长度为 {len(v_keys)}，必须包含至少 2 个视频"

        # 3.3 检查 key 是否都在 valid_video_list 中 (防止模型臆造 ID)
        for key in v_keys:
            if key not in valid_keys_set:
                return False, f"第 {index + 1} 个方案包含无效的视频 ID：'{key}' (不在原始数据中)"

        # 4. 检查字段内容是否为空 (可选，但建议加上)
        if not plan.get('video_theme') or not isinstance(plan.get('video_theme'), str):
            return False, f"第 {index + 1} 个方案 'new_video_theme' 为空或类型错误"

        # 5. 检查 score 是否能转换为 float
        score_val = plan.get('score')
        try:
            float(score_val)
        except (ValueError, TypeError):
            return False, f"第 {index + 1} 个方案 'score' 格式错误：'{score_val}' 无法转换为浮点数"

    # 所有检查通过
    return True, ""


def gen_hot_video_llm(video_info, hot_video=None):

    if hot_video:
        PROMPT_FILE_PATH = './prompt/挖掘热门视频.txt'
    else:
        PROMPT_FILE_PATH = './prompt/挖掘热门视频无热榜.txt'
    last_prompt = """# Action:
请根据上述所有指令和数据，进行深度分析并输出最终结果：
"""
    base_prompt = read_file_to_str(PROMPT_FILE_PATH)
    if hot_video:
        base_prompt = f"{base_prompt}\n当前热门视频主题 (Trending Themes)如下：\n{hot_video}\n\n"

    full_prompt = f"{base_prompt}\n视频素材信息 (Video Materials)如下：\n{video_info}\n\n{last_prompt}"
    model_name = "gemini-2.5-pro"
    # model_name = "gemini-3-pro-preview"
    max_retries = 3
    video_content_plans = []
    for attempt in range(1, max_retries + 1):
        try:
            print(f"尝试第 {attempt} 次生成视频内容计划 素材长度为{len(video_info)} {PROMPT_FILE_PATH}...")
            raw = ask_gemini(full_prompt, model_name=model_name)

            # raw = get_llm_content(prompt=full_prompt, model_name=model_name)
            video_content_plans = string_to_object(raw)
            valid_video_list = video_info.keys()
            is_valid, error_message = check_video_content_plan(video_content_plans, valid_video_list)
            if not is_valid:
                raise ValueError(f"第 {attempt} 次尝试: LLM 返回的数据格式或内容无效: {error_message}")

            for plan in video_content_plans:
                plan['dig_time'] = int(time.time())
            return video_content_plans, full_prompt
        except Exception as e:
            print(f"第 {attempt} 次尝试: 生成视频内容计划时出错: {e}")
            time.sleep(2 ** attempt)
    return video_content_plans, full_prompt


def get_target_tags(manager, task_info):
    """
    生成最终的目标标签
    :param manager:
    :param task_info:
    :return:
    """
    upload_info_list = task_info.get('upload_info', [])
    video_type, target_tags = gen_true_type_and_tags(upload_info_list)
    video_id_list = task_info.get('video_id_list', [])
    video_info_list = manager.find_materials_by_ids(video_id_list)
    for video_info in video_info_list:
        logical_scene_info = video_info.get('logical_scene_info', {})
        tags_info = get_tags_info(logical_scene_info)
        for tag, weight in tags_info.items():
            if tag in target_tags:
                target_tags[tag] += weight
            else:
                target_tags[tag] = weight
    return target_tags


def get_need_dig_video_list():
    """
    获取所有待挖掘的视频，每个题材至少会有一个视频
    :return:
    """
    all_dig_video_list = []
    statistic_play_info = read_json(STATISTIC_PLAY_COUNT_FILE)
    good_video_list = statistic_play_info.get('good_video_list', [])
    good_tags_info = statistic_play_info.get('good_tags_info', {})
    good_video_list = sorted(good_video_list, key=lambda x: (x.get('final_score', 0)), reverse=True)
    user_config = read_json(r'W:\project\python_project\auto_video\config\user_config.json')
    video_type_count_map = {}
    for target_video_type in ['sport', 'game', 'fun']:
        user_type_info = user_config.get('user_type_info', {})
        target_user_list = user_type_info.get(target_video_type, [])
        filter_good_video_list = [task_info for task_info in good_video_list if task_info.get('userName') in target_user_list]
        if len(filter_good_video_list) == 0:
            print(f"未找到符合条件的热门视频，跳过本次挖掘。{target_video_type}")
            continue

        final_good_video_list = [filter_good_video_list[0]]
        min_score = 1000
        for task_info in filter_good_video_list:
            final_score = task_info.get('final_score', 0)
            if final_score >= min_score:
                final_good_video_list.append(task_info)
        video_type_count_map[target_video_type] = len(final_good_video_list)
        all_dig_video_list.extend(final_good_video_list)
    formatted_map = json.dumps(video_type_count_map, ensure_ascii=False, indent=4)
    print(f"本次挖掘总共涉及热门视频数量: {len(all_dig_video_list)} ，各题材视频数量分布:\n {formatted_map} 当前时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    return all_dig_video_list, good_tags_info, good_video_list


def get_target_video(all_video_info, target_tags, target_video_type, no_asr=False, top_n=100):
    """
    获取本次挖掘最终的素材列表
    :return:
    """
    filter_all_video_info = {vid: info for vid, info in all_video_info.items() if info.get('video_type') == target_video_type}
    if target_video_type == 'game' and no_asr:
        delete_keys = []
        for vid, info in filter_all_video_info.items():
            owner_asr_info = info.get('owner_asr_info', [])
            if is_contain_owner_speaker(owner_asr_info):
                delete_keys.append(vid)

        for vid in delete_keys:
            del filter_all_video_info[vid]

    sorted_video_info = process_and_sort_video_info(filter_all_video_info, target_tags)
    top_video_info = dict(list(sorted_video_info.items())[:top_n])
    good_video_info = {}
    min_match_score = 1
    for vid, info in top_video_info.items():
        if info.get('match_score', 0) > 1:
            good_video_info[vid] = info
            min_match_score = info.get('match_score', 0)

    final_good_video_list = good_video_info.copy()

    # 如果good_video_info超过50就随机选择50
    if len(final_good_video_list) > top_n - 50:
        selected_keys = random.sample(list(final_good_video_list.keys()), top_n - 50)
        final_good_video_list = {key: final_good_video_list[key] for key in selected_keys}

    print(f"本次挖掘符合条件的素材视频数量: {len(final_good_video_list)}，过滤前的数量为{len(filter_all_video_info)} 前{top_n}数量为： {len(good_video_info)} 最低匹配得分: {min_match_score}，当前时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

    return final_good_video_list

def check_need_dig(exist_video_plan_info, hot_video, max_dig_count=5):
    """
    检查是否需要进行视频方案挖掘
    :return:
    """
    if hot_video not in exist_video_plan_info:
        exist_video_plan_info[hot_video] = []
    # 判断exist_video_plan_info[hot_video]中 score 大于95的方案数量是否大于等于10
    high_score_plan_count = sum(1 for plan in exist_video_plan_info[hot_video] if plan.get('score', 0) >= 95)
    print(f"当前热门视频主题: {hot_video} 已挖掘数量{len(exist_video_plan_info[hot_video])} ，高分方案数量: {high_score_plan_count}，当前时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    if high_score_plan_count >= max_dig_count:
        return False
    return True


def update_exist_dig_data(exist_video_plan_info, good_video_list):
    """
    根据good_video_list更新exist_video_plan_info已有话题的得分
    :param exist_video_plan_info:
    :param good_video_list:
    :return:
    """
    hot_topic_score_map = {}

    video_score_info = {}
    block_video_id_list = read_json(BLOCK_VIDEO_ID_FILE)

    # 【新增】打印黑名单长度，并转为set提高查找速度
    print(f"黑名单长度: {len(block_video_id_list)}")
    block_video_id_set = set(block_video_id_list)
    removed_plan_count = 0

    for video_info in good_video_list:
        upload_params = video_info.get('upload_params', {})
        hot_video = upload_params.get('title', '变得有吸引力一点')
        final_score = video_info.get('final_score', 0)
        hot_topic_score_map[hot_video] = final_score
        video_id_list = video_info.get('video_id_list', [])
        for video_id in video_id_list:
            if video_id not in video_score_info:
                video_score_info[video_id] = []
            video_score_info[video_id].append(final_score)

    # 最终生成一个video_id的平均得分的map
    video_id_avg_score_map = {}
    for video_id, score_list in video_score_info.items():
        avg_score = sum(score_list) / len(score_list)
        video_id_avg_score_map[video_id] = avg_score

    for hot_topic, video_info_list in exist_video_plan_info.items():
        # 【新增】黑名单过滤逻辑
        # 创建一个新的列表来存储未被剔除的plan
        valid_plans = []
        for plan in video_info_list:
            video_id_list = plan.get('video_id_list', [])
            # 检查是否有任何id在黑名单中
            is_blocked = False
            for vid in video_id_list:
                if vid in block_video_id_set:
                    is_blocked = True
                    break

            if is_blocked:
                removed_plan_count += 1
            else:
                valid_plans.append(plan)

        # 更新字典中的列表为过滤后的列表，并更新局部变量供后续逻辑使用
        exist_video_plan_info[hot_topic] = valid_plans
        video_info_list = valid_plans

        if hot_topic in hot_topic_score_map:
            final_score = hot_topic_score_map[hot_topic]
            for plan in video_info_list:
                plan['final_score'] = final_score * plan.get('score', 0) / 100 * 0.5
                plan['update_time'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        else:
            for plan in video_info_list:
                final_score = 100
                video_id_list = plan.get('video_id_list', [])
                temp_score_list = []
                for video_id in video_id_list:
                    temp_score_list.append(video_id_avg_score_map.get(video_id, 0))
                # 计算平均值
                if len(temp_score_list) > 0:
                    final_score = sum(temp_score_list) / len(temp_score_list) + final_score

                plan['final_score'] = final_score * plan.get('score', 0) / 100 * 0.8
                plan['update_time'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

    # 【新增】打印剔除的plan个数
    print(f"已经剔除的plan个数: {removed_plan_count}")

    return exist_video_plan_info, video_id_avg_score_map


def archive_outdated_plans(exist_info, archive_info, days=7):
    """
    功能：
    1. 将 exist_info 中超过 days 天的数据移入 archive_info。
    2. 自动清理 exist_info 中为空的主题（Key），防止空列表堆积。
    3. 打印高信息量日志：归档数、剩余数、清理掉的空主题数。
    """
    current_time = int(time.time())
    expire_threshold = current_time - (days * 24 * 3600)

    # 统计计数器
    stat_archived_count = 0  # 归档条数
    stat_remaining_count = 0  # 剩余条数
    stat_cleaned_topics = 0  # 被删除的空主题数（原本空的 + 归档后空的）

    start_cpu_time = time.time()

    # 获取所有主题的列表，防止在遍历字典时修改字典大小报错
    all_topics = list(exist_info.keys())
    initial_topic_count = len(all_topics)

    for topic in all_topics:
        plans = exist_info.get(topic, [])
        active_plans = []  # 留下的（7天内）
        expired_plans = []  # 要走的（7天前）

        # 分拣数据
        for plan in plans:
            # 获取时间戳，如果没有时间戳，默认视为当前时间（不归档）
            plan_ts = plan.get('timestamp', current_time)

            if plan_ts < expire_threshold:
                expired_plans.append(plan)
            else:
                active_plans.append(plan)

        # --- 1. 处理归档数据 ---
        if expired_plans:
            stat_archived_count += len(expired_plans)

            if topic not in archive_info:
                archive_info[topic] = []
            archive_info[topic].extend(expired_plans)

        # --- 2. 处理剩余数据 & 清理空Key ---
        if active_plans:
            # 如果还有活着的方案，更新回去
            exist_info[topic] = active_plans
            stat_remaining_count += len(active_plans)
        else:
            # 【关键】如果列表为空（无论是原本空还是归档后空），直接删除该 Key
            del exist_info[topic]
            stat_cleaned_topics += 1

    # --- 高信息量日志输出 ---
    cost_time = time.time() - start_cpu_time
    current_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

    # 只要有变动（归档了数据 OR 清理了空主题）就打印详细日志
    if stat_archived_count > 0 or stat_cleaned_topics > 0:
        print(f"【归档清理报告】{current_time_str} | "
              f"归档方案: {stat_archived_count}条 | "
              f"清理空主题: {stat_cleaned_topics}个 | "
              f"当前剩余: {stat_remaining_count}条 (共{len(exist_info)}个主题) | "
              f"耗时: {cost_time:.4f}s")
    else:
        # 如果没有任何变动，打印简报
        print(f"【归档扫描】{current_time_str} | 无过期数据且无空主题 | "
              f"当前保持: {stat_remaining_count}条 (共{len(exist_info)}个主题)")

    return exist_info, archive_info

def find_good_plan(manager):
    """
    通过已有素材找到合适的更加好的视频方案来制作视频
    :return:
    """
    # 获取本轮待挖掘的视频
    all_dig_video_list, good_tags_info, good_video_list = get_need_dig_video_list()
    # 判断是否有新的话题，有的化就需要重新拉取数据
    is_need_refresh = False
    exist_video_plan_info = read_json(DIG_HOT_VIDEO_PLAN_FILE)
    archive_video_plan_info = read_json(DIG_HOT_VIDEO_PLAN_ARCHIVE_FILE)

    # =============== 插入点开始 ===============
    # 执行归档：这一步会把 exist_video_plan_info 里旧的移走，剩下新的
    exist_video_plan_info, archive_video_plan_info = archive_outdated_plans(
        exist_video_plan_info,
        archive_video_plan_info,
        days=7
    )

    # 务必在这里保存一次，确保归档操作被持久化
    # 这样后续逻辑拿到的 exist_video_plan_info 就是干净的 7 天内数据
    save_json(DIG_HOT_VIDEO_PLAN_ARCHIVE_FILE, archive_video_plan_info)
    save_json(DIG_HOT_VIDEO_PLAN_FILE, exist_video_plan_info)


    exist_video_plan_info, video_id_avg_score_map = update_exist_dig_data(exist_video_plan_info, good_video_list)
    save_json(DIG_HOT_VIDEO_PLAN_FILE, exist_video_plan_info)

    for selected_video_info in all_dig_video_list:
        upload_params = selected_video_info.get('upload_params', {})
        hot_video = upload_params.get('title', '变得有吸引力一点')
        if hot_video not in exist_video_plan_info:
            is_need_refresh = True
            break

    # 获得素材库数据
    all_video_info = query_all_material_videos(manager, is_need_refresh)


    # 依次的进行挖掘
    for selected_video_info in all_dig_video_list:
        user_name = selected_video_info.get('userName', '')
        target_video_type = get_user_type(user_name)
        play_comment_info_list = selected_video_info.get('play_comment_info_list')
        upload_params = selected_video_info.get('upload_params', {})
        hot_video = upload_params.get('title', '变得有吸引力一点')
        final_score = selected_video_info.get('final_score', 0)
        start_time = time.time()

        check_need_dig_result = check_need_dig(exist_video_plan_info, hot_video)
        exist_count = len(exist_video_plan_info[hot_video])
        if check_need_dig_result:
            # 选择出素材组
            target_tags = get_target_tags(manager, selected_video_info)

            final_good_video_list = get_target_video(all_video_info, target_tags, target_video_type)
            video_data = build_prompt_data(final_good_video_list, target_video_type)
            print(f"符合条件的热门视频数量: {len(final_good_video_list)}，当前热门视频主题: {hot_video} 已挖掘数量{len(exist_video_plan_info[hot_video])} ，final_score: {final_score} 数据为 {play_comment_info_list[-1]} 素材视频数量: {len(video_data)}，当前时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
            video_content_plans, full_prompt = gen_hot_video_llm(video_data, hot_video)
            exist_video_plan_info[hot_video].extend(video_content_plans)

        for plan in exist_video_plan_info[hot_video]:
            creative_guidance = f"视频主题: {plan.get("video_theme")}, {plan.get("story_outline")}"
            plan['video_type'] = target_video_type
            plan['final_score'] = final_score * plan.get('score', 0) / 100 * 0.5
            plan['dig_type'] = "exist_video_dig"
            plan['timestamp'] = int(time.time())
            plan['update_time'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            plan['creative_guidance'] = creative_guidance


        save_json(DIG_HOT_VIDEO_PLAN_FILE, exist_video_plan_info)
        print(f"完成视频方案挖掘，当前热门视频主题: {hot_video}，新挖掘数量: {len(exist_video_plan_info[hot_video]) - exist_count}，耗时: {time.time() - start_time:.2f} 秒 当前时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n\n")

    # free_dig_info_path = r'W:\project\python_project\auto_video\config\free_dig_info.json'
    # free_dig_info = read_json(free_dig_info_path)
    # 进行自由挖掘
    for target_video_type, target_tags in good_tags_info.items():
        if target_video_type != 'game':
            continue
        final_good_video_list = get_target_video(all_video_info, target_tags, target_video_type, no_asr=True, top_n=150)
        video_data = build_prompt_data(final_good_video_list, target_video_type)
        print(f"符合条件的热门视频数量: {len(final_good_video_list)}，当前自由挖掘类型: {target_video_type}  素材视频数量: {len(video_data)}，当前时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
        video_content_plans, full_prompt = gen_hot_video_llm(video_data, None)
        timestamp = int(time.time())
        # if len(video_content_plans) > 0:
        #     free_dig_info[f"{timestamp}"] = {
        #         'video_content_plans':video_content_plans,
        #         'full_prompt': full_prompt,
        #
        #     }
        #     save_json(free_dig_info_path, free_dig_info)
        for plan_info in video_content_plans:
            hot_video = plan_info.get('video_theme', '有趣的视频')
            if hot_video not in exist_video_plan_info:
                exist_video_plan_info[hot_video] = []
            exist_video_plan_info[hot_video].append(plan_info)
            final_score = 100
            for plan in exist_video_plan_info[hot_video]:

                video_id_list = plan.get('video_id_list', [])
                temp_score_list = []
                for video_id in video_id_list:
                    temp_score_list.append(video_id_avg_score_map.get(video_id, 0))
                # 计算平均值
                if len(temp_score_list) > 0:
                    final_score = sum(temp_score_list) / len(temp_score_list) + final_score
                creative_guidance = f"视频主题: {plan.get("video_theme")}, {plan.get("story_outline")}"
                plan['video_type'] = target_video_type
                plan['final_score'] = final_score * plan.get('score', 0) / 100 * 0.8
                plan['dig_type'] = "free_dig_new"
                plan['timestamp'] = timestamp
                plan['update_time'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
                plan['creative_guidance'] = creative_guidance
        save_json(DIG_HOT_VIDEO_PLAN_FILE, exist_video_plan_info)

        print(f"完成视频方案挖掘，当前自由挖掘类型: {target_video_type}，新挖掘数量: {len(video_content_plans)}，当前时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n\n")



if __name__ == '__main__':
    # time.sleep(3600 * 3)
    mongo_base_instance = gen_db_object()
    manager = MongoManager(mongo_base_instance)
    while True:
        try:
            find_good_plan(manager)
            time.sleep(1)
        except Exception as e:
            traceback.print_exc()
            print(f"挖掘视频方案时出错: {e}")
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
import os
import random
import time
import traceback

from application.video_common_config import ALL_MATERIAL_VIDEO_INFO_PATH, BLOCK_VIDEO_BVID_FILE, get_tags_info, \
    DIG_HOT_VIDEO_PLAN_FILE, STATISTIC_PLAY_COUNT_FILE
from utils.common_utils import read_json, save_json, has_long_common_substring, read_file_to_str, string_to_object, \
    gen_true_type_and_tags
from utils.gemini_cli import ask_gemini
from utils.mongo_base import gen_db_object
from utils.mongo_manager import MongoManager

NEED_REFRESH = False


def query_all_material_videos():
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
        if time.time() - modify_time < 86400 and not NEED_REFRESH:
            print(
                f"all_need_plan_video_info.json 缓存文件存在且在一天之内，直接读取。当前数据量: {len(local_material_video_info)}, 耗时: {time.time() - start_time:.4f}s")  # [修改] 增加耗时和数据量打印
            return local_material_video_info

    mongo_base_instance = gen_db_object()
    manager = MongoManager(mongo_base_instance)
    query = {}

    all_material_list = manager.find_by_custom_query(manager.materials_collection, query)
    raw_db_count = len(all_material_list)  # [新增] 记录数据库原始查询数量

    for video_info in all_material_list:
        video_id = video_info['video_id']
        logical_scene_info = video_info.get('logical_scene_info', {})
        tags_info = get_tags_info(logical_scene_info)
        if tags_info:
            temp_dict = {}
            temp_dict['tags_info'] = tags_info
            temp_dict['logical_scene_info'] = logical_scene_info
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

        # --- 第一部分：基于 tags_info 的加权匹配 ---
        video_tags_info = info.get('tags_info', {})

        # 只有当 tags_info 不为空时才进行这部分计算
        if video_tags_info:
            for v_tag, v_weight in video_tags_info.items():
                for t_tag, t_weight in target_tags_info.items():
                    # 调用外部匹配函数
                    if has_long_common_substring(v_tag, t_tag):
                        # 双方都有权重，乘积累加
                        total_score += v_weight * t_weight

        # --- 第二部分：基于 info 全文字符串的匹配 ---
        # 将整个字典转为字符串 (包含了 tags_info, logical_scene_info 等所有字段)
        info_str = str(info)

        for t_tag, t_weight in target_tags_info.items():
            # 将“整个视频信息字符串”作为一个大字符串，去和目标标签对比
            # 如果匹配成功，说明该视频的某个角落（摘要、场景、或者标签里）提到了这个词
            if has_long_common_substring(info_str, t_tag):
                # 因为全文匹配没有对应的视频侧权重，这里加上目标标签原本的权重
                total_score += t_weight * t_weight

        # 将最终计算的总分写入字典
        info['match_score'] = total_score

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

def build_prompt_data(video_info):
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

    PROMPT_FILE_PATH = './prompt/挖掘热门视频.txt'
    last_prompt = """# Action:
请根据上述所有指令和数据，进行深度分析并输出最终结果：
"""
    base_prompt = read_file_to_str(PROMPT_FILE_PATH)
    if hot_video:
        base_prompt = f"{base_prompt}\n当前热门视频主题 (Trending Themes)如下：\n{hot_video}\n\n"

    full_prompt = f"{base_prompt}\n视频素材信息 (Video Materials)如下：\n{video_info}\n\n{last_prompt}"
    model_name = "gemini-2.5-pro"
    model_name = "gemini-3-pro-preview"
    max_retries = 3
    video_content_plans = []
    for attempt in range(1, max_retries + 1):
        try:
            print(f"尝试第 {attempt} 次生成视频内容计划 素材长度为{len(video_info)}...")
            raw = ask_gemini(full_prompt, model_name=model_name)

            # raw = get_llm_content(prompt=full_prompt, model_name=model_name)
            video_content_plans = string_to_object(raw)
            valid_video_list = video_info.keys()
            is_valid, error_message = check_video_content_plan(video_content_plans, valid_video_list)
            if not is_valid:
                raise ValueError(f"第 {attempt} 次尝试: LLM 返回的数据格式或内容无效: {error_message}")
            return video_content_plans
        except Exception as e:
            print(f"第 {attempt} 次尝试: 生成视频内容计划时出错: {e}")
            time.sleep(2 ** attempt)
    return video_content_plans


def find_good_plan():
    """
    通过已有素材找到合适的更加好的视频方案来制作视频
    :return:
    """
    statistic_play_info = read_json(STATISTIC_PLAY_COUNT_FILE)
    good_video_list = statistic_play_info.get('good_video_list', [])
    good_video_list.sort(key=lambda x: len(x.get("choose_reason", [])), reverse=True)
    good_video_list = sorted(good_video_list, key=lambda x: (x.get('final_score', 0)), reverse=True)
    target_video_type = random.choice(['game', 'fun'])

    user_config = read_json(r'W:\project\python_project\auto_video\config\user_config.json')
    user_type_info = user_config.get('user_type_info', {})
    target_user_list = user_type_info.get(target_video_type, [])

    filter_good_video_list = [task_info for task_info in good_video_list if task_info.get('userName') in target_user_list]
    if not filter_good_video_list:
        print(f"未找到符合条件的热门视频，跳过本次挖掘。{target_video_type}")
        return
    final_good_video_list = [filter_good_video_list[0]]
    min_score = 500
    max_count = 10
    for task_info in filter_good_video_list:
        final_score = task_info.get('final_score', 0)
        if final_score >= min_score:
            final_good_video_list.append(task_info)
        if len(final_good_video_list) >= max_count:
            break

    selected_video_info = random.choice(final_good_video_list)
    play_comment_info_list = selected_video_info.get('play_comment_info_list')

    upload_params = selected_video_info.get('upload_params', {})
    hot_video = upload_params.get('title', '变得有吸引力一点')
    final_score = selected_video_info.get('final_score', 0)

    upload_info_list = selected_video_info.get('upload_info', [])
    video_type, target_tags = gen_true_type_and_tags(upload_info_list)


    start_time = time.time()
    exist_video_plan_info = read_json(DIG_HOT_VIDEO_PLAN_FILE)
    all_video_info = query_all_material_videos()
    # target_tags = gen_target_tags()

    sorted_video_info = process_and_sort_video_info(all_video_info, target_tags)
    # 选择匹配得分前100的视频素材
    top_n = 100
    top_video_info = dict(list(sorted_video_info.items())[:top_n])
    good_video_info = {}
    for vid, info in top_video_info.items():
        if info.get('match_score', 0) >= 1:
            good_video_info[vid] = info


    # 如果good_video_info超过50就随机选择50
    if len(good_video_info) > 50:
        selected_keys = random.sample(list(good_video_info.keys()), 50)
        good_video_info = {key: good_video_info[key] for key in selected_keys}

    video_data = build_prompt_data(good_video_info)

    print(f"符合条件的热门视频数量: {len(final_good_video_list)}，当前热门视频主题: {hot_video}  ，final_score: {final_score} 数据为 {play_comment_info_list[-1]} 素材视频数量: {len(video_data)}，当前时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    video_content_plans = gen_hot_video_llm(video_data, hot_video)
    if hot_video not in exist_video_plan_info:
        exist_video_plan_info[hot_video] = []
    exist_video_plan_info[hot_video].extend(video_content_plans)

    for plan in video_content_plans:
        plan['user_type_info'] = target_video_type
        plan['final_score'] = final_score
        plan['update_time'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

    save_json(DIG_HOT_VIDEO_PLAN_FILE, exist_video_plan_info)
    print(f"完成视频方案挖掘，当前热门视频主题: {hot_video}，共挖掘出 {len(video_content_plans)} 个新方案，耗时: {time.time() - start_time:.2f} 秒 当前时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n\n")



if __name__ == '__main__':

    while True:
        try:
            find_good_plan()
        except Exception as e:
            traceback.print_exc()
            print(f"挖掘视频方案时出错: {e}")
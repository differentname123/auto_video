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
import time

from application.video_common_config import ALL_MATERIAL_VIDEO_INFO_PATH, BLOCK_VIDEO_BVID_FILE, get_tags_info
from utils.common_utils import read_json, save_json
from utils.mongo_base import gen_db_object
from utils.mongo_manager import MongoManager

NEED_REFRESH = True

def query_all_material_videos():
    """
    查询所有的素材视频，已剔除黑名单素材
    :return:
    """
    global NEED_REFRESH
    local_material_video_info = read_json(ALL_MATERIAL_VIDEO_INFO_PATH)
    # 判断ALL_MATERIAL_VIDEO_INFO_PATH这个文件上次修改时间是否超过1天，超过1天则重新从数据库中查询
    if os.path.exists(ALL_MATERIAL_VIDEO_INFO_PATH):
        modify_time = os.path.getmtime(ALL_MATERIAL_VIDEO_INFO_PATH)
        # 86400秒 = 1天
        if time.time() - modify_time < 86400 and not NEED_REFRESH:
            print("all_need_plan_video_info.json 缓存文件存在且在一天之内，直接读取。")
            return local_material_video_info


    mongo_base_instance = gen_db_object()
    manager = MongoManager(mongo_base_instance)
    query = {}

    all_material_list = manager.find_by_custom_query(manager.materials_collection, query)
    for video_info in all_material_list:
        video_id = video_info['video_id']
        logical_scene_info = video_info.get('logical_scene_info', {})
        tags_info = get_tags_info(logical_scene_info)
        if tags_info:
            video_info['tags_info'] = tags_info
            local_material_video_info[video_id] = video_info

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


    # 删除在block_video_id_list中的视频
    for block_video_id in block_video_id_list:
        if block_video_id in local_material_video_info:
            del local_material_video_info[block_video_id]


    save_json(ALL_MATERIAL_VIDEO_INFO_PATH, local_material_video_info)
    NEED_REFRESH = False
    return local_material_video_info

def find_good_plan():
    """
    通过已有素材找到合适的更加好的视频方案来制作视频
    :return:
    """
    all_video_info = query_all_material_videos()


if __name__ == '__main__':
    query_all_material_videos()
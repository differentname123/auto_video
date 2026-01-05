# -- coding: utf-8 --
""":authors:
    zhuxiaohu
:create_date:
    2025/12/14 18:39
:last_date:
    2025/12/14 18:39
:description:
    进行视频的制作以及投稿
"""
import time

from application.video_common_config import TaskStatus
from utils.mongo_base import gen_db_object
from utils.mongo_manager import MongoManager


def auto_upload(manager):
    """
    进行单次循环的投稿
    :return:
    """
    tasks_to_upload = manager.find_tasks_by_status([TaskStatus.PLAN_GENERATED])


    pass



if __name__ == "__main__":
    mongo_base_instance = gen_db_object()
    manager = MongoManager(mongo_base_instance)
    while True:
        auto_upload(manager)
        time.sleep(60)  # 每分钟运行一次
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
import traceback
from datetime import datetime
from collections import defaultdict

from application.process_video import process_single_task, query_need_process_tasks
from application.video_common_config import TaskStatus, ERROR_STATUS, check_failure_details, build_task_video_paths
from utils.common_utils import read_json, is_valid_target_file_simple
from utils.mongo_base import gen_db_object
from utils.mongo_manager import MongoManager


def gen_user_upload_info(uploaded_tasks_today):
    """
    通过今日投稿的任务生成用户投稿的信息
    """
    # 定义默认值结构
    user_upload_info = defaultdict(lambda: {'count': 0, 'latest_upload_time': datetime.min})

    for task in uploaded_tasks_today:
        user_name = task['userName']
        upload_time = task['upload_time']

        # 更新数据
        info = user_upload_info[user_name]
        info['count'] += 1
        # 比较并保留较大的时间
        if upload_time > info['latest_upload_time']:
            info['latest_upload_time'] = upload_time

    return dict(user_upload_info)  # 转回普通字典返回


def sort_tasks(tasks_list, user_info_map):
    """
    对任务列表进行排序
    规则: count(asc) -> schedule_date的日期(asc) -> update_time(asc)
    """

    def get_sort_key(task):
        # 1. 获取 userName
        # 假设 task 是对象，使用 task.userName
        # 如果 task 是字典，请改为 task['userName']
        user_name = task.userName

        # 2. 获取 count
        # 从 map 中获取 user 对象
        user_obj = user_info_map.get(user_name)

        if user_obj:
            # 假设 user_obj 是对象，使用 user_obj.count
            # 如果 user_obj 是字典，请改为 user_obj['count']
            count = user_obj.count
        else:
            # 如果没查询到用户信息对象，count 为 0
            count = 0

        # 3. 获取 schedule_date 并转换为只包含日期的对象
        # 假设结构为 task.creation_guidance_info.schedule_date
        schedule_dt = task.creation_guidance_info.schedule_date
        schedule_day = schedule_dt.date()  # 只取日期部分 (YYYY-MM-DD)

        # 4. 获取 update_time
        update_time = task.update_time

        # 5. 返回元组进行多级排序
        # Python 的元组比较机制会自动按照顺序比较：先比第一个元素，相同则比第二个，以此类推
        return (count, schedule_day, update_time)

    # 使用 key 进行原地排序
    tasks_list.sort(key=get_sort_key)

    return tasks_list

def gen_video_and_upload(task_info, manager):
    """
    生成视频并进行投稿
    :param task_info:
    :return:
    """
    failure_details = {}
    upload_params = {}
    all_task_video_path_info = build_task_video_paths(task_info)
    final_output_path = all_task_video_path_info['final_output_path']
    if not is_valid_target_file_simple(final_output_path):
        failure_details, video_info_dict = process_single_task(task_info, manager, gen_video=True)
    if check_failure_details(failure_details):
        return failure_details, upload_params

    # 继续加工视频，主要是进行水印的增加以及ending的添加






def auto_upload(manager):
    """
    进行单次循环的投稿
    :return:
    """
    for i in range(10):
        start_time = time.time()
        tobe_upload_video_info_file = r'W:\project\python_project\auto_video\config\tobe_upload_video_info.json'
        tobe_upload_video_info = read_json(tobe_upload_video_info_file)
        tasks_to_upload = manager.find_tasks_by_status([TaskStatus.PLAN_GENERATED])
        print(f"找到 {len(tasks_to_upload)} 个待投稿任务，开始处理...耗时 {time.time() - start_time:.2f} 秒")
        start_time = time.time()

        tasks_to_process = query_need_process_tasks()
        print(f"找到 {len(tasks_to_process)} 个待处理任务，开始处理...耗时 {time.time() - start_time:.2f} 秒")

    return

    now = datetime.now()
    today_midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    # 查询今日已投稿的任务
    uploaded_tasks_today = manager.find_tasks_after_time_with_status(today_midnight, [TaskStatus.UPLOADED])
    user_upload_info = gen_user_upload_info(uploaded_tasks_today)

    sort_tasks_to_upload = sort_tasks(tasks_to_upload, tobe_upload_video_info)

    for task_info in sort_tasks_to_upload:

        try:
            failure_details, video_info_dict = process_single_task(task_info, manager, gen_video=True)
        except Exception as e:
            traceback.print_exc()
            error_info = f"严重错误: 处理任务 {task_info.get('_id', 'N/A')} 时发生未知异常: {str(e)}"
            print(error_info)
            failure_details[str(task_info.get('_id', 'N/A'))] = {
                "error_info": error_info,
                "error_level": ERROR_STATUS.CRITICAL
            }
            # 原代码在循环中使用了 continue，此处函数执行完异常处理后会自动进入 finally，效果一致
        finally:
            if check_failure_details(failure_details):
                failed_count = task_info.get('failed_count', 0)
                task_info['failed_count'] = failed_count + 1
                task_info['status'] = TaskStatus.FAILED
            else:
                # task_info['status'] = TaskStatus.COMPLETED
                pass

            task_info['failure_details'] = str(failure_details)
            manager.upsert_tasks([task_info])



if __name__ == "__main__":
    mongo_base_instance = gen_db_object()
    manager = MongoManager(mongo_base_instance)
    while True:
        auto_upload(manager)
        time.sleep(60)  # 每分钟运行一次
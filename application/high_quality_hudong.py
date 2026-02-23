#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import random
import traceback
import time
import logging
import os
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from datetime import datetime, timedelta
from types import SimpleNamespace

from application.video_common_config import TaskStatus, USER_BVID_FILE, ALL_BVID_FILE, COMMENTER_USAFE_FILE, \
    STATISTIC_PLAY_COUNT_FILE, BLOCK_VIDEO_BVID_FILE
from utils.bilibili.bili_utils import update_bili_user_sign, block_all_author
from utils.bilibili.bilibili_uploader import send_bilibili_dm_command
from utils.bilibili.comment import BilibiliCommenter, get_bilibili_archives
from utils.bilibili.get_danmu import gen_proper_comment
from utils.bilibili.watch_video import watch_video
from utils.common_utils import get_config, read_json, init_config, save_json, get_simple_play_distribution, \
    calculate_averages, gen_true_type_and_tags, get_user_type
from utils.mongo_base import gen_db_object
from utils.mongo_manager import MongoManager

# --- 全局配置 ---
total_cookie = get_config("dahao_bilibili_total_cookie")
csrf_token = get_config("dahao_bilibili_csrf_token")


def path_exists(path) -> bool:
    """
    判断输入的路径字符串是否存在。
    """
    if not isinstance(path, str):
        return False
    stripped = path.strip()
    if not stripped:
        return False
    return os.path.exists(stripped)


def post_comments_once(commenter_list,
                       comment_list,
                       bvid,
                       max_success_comment_count,
                       comment_used_list,
                       path_exists,
                       max_workers=5,
                       jitter=(0.4, 1.0)):
    """
    最终修订版V3：
    1. 使用 futures.wait() 实现可靠的全局超时。
    2. 修复了 comment_used_list 的同步BUG，只记录真正成功的评论。
    3. 将 jitter 延迟放回 worker 线程以实现并发延迟。
    4. 确保返回的 success_count 是在超时前确定的值。
    """
    # --- 1. 准备工作：分配评论任务 ---
    random.shuffle(commenter_list)
    selected = commenter_list[:max_success_comment_count]

    used_lock = threading.Lock()
    successful_texts_lock = threading.Lock()
    used_texts = set(comment_used_list)
    successful_texts = []

    assignments = []
    for c in selected:
        assigned = None
        for detail in comment_list:
            text = detail[0] if detail and len(detail) > 0 else None
            if not text or len(text) <= 1:
                continue

            with used_lock:
                if text in used_texts:
                    continue
                used_texts.add(text)
                assigned = detail
                break

        if assigned:
            assignments.append((c, assigned))
        else:
            break

    if not assignments:
        print("没有可分配的评论或 commenter，退出。")
        return 0

    # --- 2. Worker 函数定义 ---
    def worker(pair):
        time.sleep(random.uniform(*jitter))

        commenter, detail = pair
        text = detail[0]
        image_path = detail[2] if len(detail) > 2 else None

        try:
            if image_path and path_exists(image_path):
                rpid = commenter.post_comment(bvid, text, 1, like_video=True, image_path=image_path,
                                              forward_to_dynamic=False)
            else:
                rpid = commenter.post_comment(bvid, text, 1, like_video=True, forward_to_dynamic=False)

            if rpid:
                with successful_texts_lock:
                    successful_texts.append(text)
                name = commenter.all_params.get('name', 'unknown')
                print(f"[评论成功] by {name} rpid:{rpid}: {text}")
                return True
            else:
                print(f"[评论失败] by {getattr(commenter, 'name', 'unknown')} (接口返回): {text}")
                return False

        except Exception as e:
            print(f"[评论异常] by {getattr(commenter, 'name', 'unknown')}: {text} -> {e}")
            return False
        finally:
            with used_lock:
                if text in used_texts:
                    used_texts.remove(text)

    # --- 3. 核心执行区域 ---
    TOTAL_TIMEOUT = 300
    executor = ThreadPoolExecutor(max_workers=min(max_workers, len(assignments)))
    future_to_info = {executor.submit(worker, a): a[1][0] for a in assignments}
    confirmed_success_count = 0

    try:
        done, not_done = wait(future_to_info.keys(), timeout=TOTAL_TIMEOUT, return_when=ALL_COMPLETED)

        for future in done:
            try:
                if future.result():
                    confirmed_success_count += 1
            except Exception:
                pass

        if not_done:
            print(f"[评论总超时] {len(not_done)} 个任务在 {TOTAL_TIMEOUT} 秒后仍未完成，将被放弃。")
            for future in not_done:
                text = future_to_info[future]
                print(f"  - 超时任务的评论: '{text[:30]}...'")
                with used_lock:
                    if text in used_texts:
                        used_texts.remove(text)

    finally:
        print(f"在超时前确认成功的评论数: {confirmed_success_count}")
        new_successes = [text for text in successful_texts if text not in comment_used_list]
        comment_used_list.extend(new_successes)
        executor.shutdown(wait=False)
        print("线程池已发出关闭信号，主流程继续。")

    return confirmed_success_count


def send_danmaku_thread_function(owner_commenter, owner_danmu_list, max_success_owner_danmu_count, bvid,
                                 owner_danmu_used_list):
    """
    这个函数包含了发送弹幕的完整逻辑，它将在一个独立的线程中被执行。
    """
    success_owner_danmu_count = 0
    if owner_commenter:
        for detail_owner_danmu in owner_danmu_list:
            if success_owner_danmu_count >= max_success_owner_danmu_count:
                print(f"线程 {threading.current_thread().name}: 已达到最大成功UP主弹幕数，停止处理。")
                break

            danmaku_time_ms = detail_owner_danmu['建议时间戳'] * 1000
            danmu_text_list = detail_owner_danmu['推荐弹幕内容']

            for danmu_text in danmu_text_list:
                if danmu_text in owner_danmu_used_list or len(danmu_text) == 0:
                    continue

                if success_owner_danmu_count >= max_success_owner_danmu_count:
                    break

                danmaku_sent = owner_commenter.send_danmaku(
                    bvid=bvid,
                    msg=danmu_text,
                    progress=danmaku_time_ms,
                    is_up=True
                )

                if danmaku_sent:
                    owner_danmu_used_list.append(danmu_text)
                    success_owner_danmu_count += 1
                    print(
                        f" [主人弹幕发送流程成功个数 {success_owner_danmu_count}] {danmu_text} BVID: {bvid} name {owner_commenter.all_params['name']}")
                    time.sleep(random.uniform(5, 10))
                else:
                    print(
                        f"{success_owner_danmu_count} 主人弹幕发送流程失败！{danmu_text} BVID: {bvid} name {owner_commenter.all_params['name']} danmaku_time_ms: {danmaku_time_ms}")
                    time.sleep(random.uniform(10, 15))

            time.sleep(random.uniform(10, 15))
    print(f"线程 {threading.current_thread().name} 完成。成功发送 UP 主弹幕数: {success_owner_danmu_count}")


def _send_danmu_worker(danmu_list, other_commenters, bvid, max_success_other_danmu_count, stop_event, result):
    try:
        random.shuffle(other_commenters)
        senders = deque(other_commenters)
        success_count = 0
        sent_texts = []

        for detail in danmu_list:
            if stop_event.is_set():
                print("send worker: 收到停止信号，退出。")
                break

            if success_count >= max_success_other_danmu_count:
                break

            danmaku_time_ms = int(detail.get('建议时间戳', 0) * 1000)
            danmu_text_list = detail.get('推荐弹幕内容', []) or []

            for text in danmu_text_list:
                if stop_event.is_set() or success_count >= max_success_other_danmu_count:
                    break
                if not text:
                    continue

                sender = senders.popleft()
                try:
                    danmaku_sent = sender.send_danmaku(
                        bvid=bvid,
                        msg=text,
                        progress=danmaku_time_ms,
                        is_up=False
                    )
                except Exception as e:
                    print("发送异常:", e)
                    danmaku_sent = False

                senders.append(sender)

                if danmaku_sent:
                    success_count += 1
                    sent_texts.append(text)
                    print(f"[成功弹幕个数 {success_count}] {text} 发送者: {sender.all_params.get('name')}")
                else:
                    print(f"[失败] {text} 发送者: {sender.all_params.get('name')}，稍后继续或跳过。")
                    time.sleep(random.uniform(5, 10))

                time.sleep(random.uniform(1, 2))

        result.success_count = success_count
        result.sent_texts = sent_texts
        print("send worker 完成。成功发送:", success_count)
    except Exception as e:
        print("worker 未捕获异常:", e)
        result.exception = e


def start_send_danmu_background(danmu_list, other_commenters, bvid, max_success_other_danmu_count, daemon=True):
    """
    启动后台线程发送弹幕（极简版）。
    """
    stop_event = threading.Event()
    result = SimpleNamespace(success_count=0, sent_texts=[])
    t = threading.Thread(
        target=_send_danmu_worker,
        args=(danmu_list, other_commenters, bvid, max_success_other_danmu_count, stop_event, result),
        daemon=daemon
    )
    t.start()
    return t, stop_event, result


def pick_commenters(commenter_map, usage_path, n=3):
    """
    从 commenter_map 中尽量均匀选 n 个账号，读取/更新 usage_path。
    """
    usage_map = {'196823511': 6, '3546972143225467': 4, '3546717871934392': 5, '3632304865937878': 2,
                 '3546970887031023': 3, '3546979686681114': 3, '3546970725550911': 3, '3632307990694238': 3}

    usage = read_json(usage_path) or {}
    usage = {str(k): int(v) for k, v in usage.items()}
    for uid in list(commenter_map.keys()):
        usage.setdefault(str(uid), 0)

    uids = list(map(str, commenter_map.keys()))
    random.shuffle(uids)
    uids.sort(key=lambda x: usage.get(x, 0))

    selected = uids[:min(n, len(uids))]
    for uid in selected:
        usage[uid] = usage.get(uid, 0) + 8 - usage_map.get(uid, 2)

    save_json(usage_path, usage)
    selected_commenter = [commenter_map[uid] for uid in selected if uid in commenter_map]
    return selected_commenter


stop_event = threading.Event()

NEED_UPDATE_SIGN = True
signatures = [
    "谢谢你这么好看还来看看我，愿你每天都被温柔对待。",
    "能遇见你真好，祝你笑口常开。",
    "你看我一眼，我就把好运给你留着。",
    "看到你真暖，愿你的每一天都晴朗。",
    "谢谢你停留，愿快乐找上门。",
    "因为有你，我的世界更亮。",
    "你这么棒，别忘了对自己好一点。",
    "感谢你的关注，愿你心想事成。",
    "谢谢你来看我，愿你夜夜好梦。",
    "你好可爱，谢谢你来，愿你事事顺心。",
    "你的出现，让我的心情变好了。",
    "你来过，我就足够幸福了。",
    "看见你就想笑，愿你永远被喜欢。",
    "谢谢你温柔以待，愿你被生活温柔相待。",
    "有你在，平凡也变有趣。",
    "遇见你是最好的巧合，祝你安好。",
    "你的微笑很暖，谢谢你停留。",
    "谢谢你把时间借给我，愿你被世界温柔以待。",
    "你在的地方就有光，愿你前路无忧。",
    "感谢今天的相遇，愿你一直好运连连。",
    "谢谢你来看看，愿所有小确幸都向你靠近。",
    "谢谢你为我点亮一眼，愿你每天被幸运宠爱。",
    "你的好看值得被世界赞美，祝你被爱包围。",
    "因为你，平淡也变成仪式感。",
    "你的出现，让我相信美好还在。",
    "谢谢你这么温柔地看我，愿你永远被温柔相待。",
    "你把好心情带来，我把祝福送给你。",
    "有你点赞真开心，愿你此刻快乐。",
    "谢谢你路过我的世界，愿你永远心平气和。"
]


def run_periodically(manager):
    while True:
        loop_start = time.time()  # 记录本轮 fun 开始时间

        stop_event.set()
        fun_thread = threading.Thread(target=fun, args=(manager,))
        fun_thread.start()
        fun_thread.join()

        elapsed = time.time() - loop_start
        remaining = max(0, 30 * 60 - elapsed)  # 剩余等待时间
        print(f"fun 执行耗时 {elapsed:.2f} 秒，等待 {remaining:.2f} 秒后再执行下一轮...")
        if remaining > 0:
            time.sleep(remaining)


def update_play_count(recent_uploaded_tasks, user_bvid_file_data):
    """
    更新播放量信息。
    """
    # 获取当前时间，需要按照可读性输出
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_bvid_map = {}
    need_update_task_list = []
    for task_info in recent_uploaded_tasks:
        upload_result = task_info.get('upload_result', {})
        bvid = upload_result.get('bvid', '')
        if bvid:
            task_bvid_map[bvid] = task_info
    count = 0
    for user_name, bvid_info_list in user_bvid_file_data.items():
        for bvid_info in bvid_info_list:
            count += 1
            bvid = bvid_info.get('bvid', '')
            if bvid in task_bvid_map:
                created = bvid_info.get('created', 0)
                task_info = task_bvid_map[bvid]
                play_count = bvid_info.get('play', 0)
                comment_count = bvid_info.get('comment', 0)
                temp_info = {
                    'play_count': play_count,
                    'comment_count': comment_count,
                }
                single_info = {
                    current_time: temp_info
                }
                play_comment_info_list = task_info.get('play_comment_info_list', [])
                play_comment_info_list.append(single_info)
                task_info['play_comment_info_list'] = play_comment_info_list
                task_info['created'] = created

                need_update_task_list.append(task_info)
    print(f"更新播放量信息，共 {len(need_update_task_list)} 个任务需要更新。 总共平台视频数: {count} 数据库最近记录数量: {len(recent_uploaded_tasks)}")
    return need_update_task_list, recent_uploaded_tasks

def set_dm_commond(total_cookie, BILI_JCT, task_info):
    """
    尝试设置投票弹幕
    :return:
    """

    try:
        upload_info_list = task_info.get("upload_info", [])
        upload_result = task_info.get("upload_result", {})
        upload_info = upload_info_list[0] if upload_info_list else {}
        if upload_info:
            opening_poll = upload_info.get("opening_poll", {})
            question = opening_poll.get("question", "视频好看吗？")[:12]
            option_a = opening_poll.get("option_a", "好看")[:6]
            option_b = opening_poll.get("option_b", "非常好看")[:6]
            bvid = upload_result.get("bvid")
            aid = upload_result.get("aid")
            send_bilibili_dm_command(total_cookie, BILI_JCT, question, 7000, option_a, option_b, bvid, aid)

    except Exception as e:
        traceback.print_exc()
        print(f"⚠️ 设置投票弹幕失败：{e}")


def process_single_hudong_task(task_info, commenter_map, uid):
    """
    进行真正的互动处理
    """
    THREAD_JOIN_TIMEOUT = 900  # 15分钟

    today = datetime.today().isoformat()
    hudong_info = task_info.get('hudong_info', {})

    bvid = task_info.get('upload_result', {}).get('bvid', '')
    if not bvid:
        print(f"❌ process_single_hudong_task: 未找到 BVID，跳过该任务。video_id_list{task_info.get('video_id_list', [])}")
        return hudong_info, True
    print(f"[{bvid}] --- process_single_video 开始 ---")

    last_processed_date = hudong_info.get('last_processed_date', '')
    if last_processed_date:
        print(f"{bvid} 跳过：该任务已有处理日期。")
        return task_info

    print(f"[{bvid}] [步骤 1/8] 调用 gen_proper_comment 获取已有互动信息...")
    result = gen_proper_comment(bvid, dont_need_comment=True)
    print(f"[{bvid}] [步骤 1/8] gen_proper_comment 调用完成。")

    exist_comment = result.get('已有评论', [])
    exist_comment_text = [comment[0] for comment in exist_comment]
    exist_danmu = result.get('已有弹幕', [])
    exist_danmu_text = [danmu[0] for danmu in exist_danmu]
    max_success_owner_danmu_count = 5
    max_success_other_danmu_count = 5

    print(f"获得到已有评论：{len(exist_comment_text)} 条，已有弹幕：{len(exist_danmu_text)} 条。| BVID: {bvid}")
    owner_commenter = commenter_map.get(uid, None)
    other_commenters = [c for k, c in commenter_map.items() if k != uid]
    share_video = hudong_info.get("share_video", False)
    triple_like_video = hudong_info.get("triple_like_video", False)

    watch_thread = None

    print(f"[{bvid}] [步骤 2/8] 检查是否需要分享和三连...")
    if not share_video or not triple_like_video:
        print(f"[{bvid}] [步骤 2a/8] 需要执行分享/三连。正在启动 watch_video 后台线程...")

        try:
            watch_thread = threading.Thread(
                target=watch_video,
                args=([bvid],)
            )
            watch_thread.start()
            print(f"[{bvid}] [步骤 2a/8] watch_video 后台线程已启动，主程序继续执行分享操作。")
        except Exception as e:
            print(f"[{bvid}] 启动 watch_video 线程失败: {e}")

        for commenter in commenter_map.values():
            name = commenter.all_params.get('name', 'unknown')
            print(f"[{bvid}] [步骤 2b/8] 用户 '{name}' 正在执行 share_video...")
            share_success = commenter.share_video(bvid=bvid)
            if share_success:
                share_video = True
            else:
                print(f"[{bvid}] 用户 '{name}' 分享操作流程失败。")
            print(f"[{bvid}] [步骤 2b/8] 用户 '{name}' share_video 调用完成。")

            print(f"[{bvid}] [步骤 2c/8] 用户 '{name}' 正在执行 triple_like_video...")
            triple_like_success = commenter.triple_like_video(bvid=bvid)
            if triple_like_success:
                triple_like_video = True
            else:
                print(f"[{bvid}] 用户 '{name}' 一键三连操作流程失败。")
            print(f"[{bvid}] [步骤 2c/8] 用户 '{name}' triple_like_video 调用完成。")

        max_success_owner_danmu_count = 20
        max_success_other_danmu_count = 30
    print(f"[{bvid}] [步骤 2/8] 分享和三连操作检查完成（观看任务可能仍在后台进行）。")

    hudong_info['share_video'] = share_video
    hudong_info['triple_like_video'] = triple_like_video
    owner_danmu_list = hudong_info.get('owner_danmu', [])
    owner_danmu_used_list = hudong_info.get('owner_danmu_used', [])
    owner_danmu_used_list.extend(exist_danmu_text)
    danmaku_thread = None

    print(f"[{bvid}] [步骤 3/8] 准备启动主人弹幕线程...")
    if owner_commenter:
        danmaku_thread = threading.Thread(
            target=send_danmaku_thread_function,
            args=(
                owner_commenter,
                owner_danmu_list,
                max_success_owner_danmu_count,
                bvid,
                owner_danmu_used_list
            )
        )
        danmaku_thread.start()
        set_dm_commond(owner_commenter.total_cookie, owner_commenter.csrf_token, task_info)
        print(f"[{bvid}] [步骤 3/8] 主人弹幕线程已启动。")
    else:
        print(f"[{bvid}] [步骤 3/8] 无主人评论者，跳过启动主人弹幕线程。")

    danmu_list = hudong_info.get('danmu_list', [])
    danmu_used_list = hudong_info.get('danmu_used', [])
    danmu_used_list.extend(exist_danmu_text)

    print(f"[{bvid}] [步骤 4/8] 准备启动其他用户弹幕线程...")
    t, stop_event, result = start_send_danmu_background(danmu_list, other_commenters, bvid,
                                                        max_success_other_danmu_count)
    print(f"[{bvid}] [步骤 4/8] 其他用户弹幕线程已启动。")

    max_success_comment_count = 5
    if uid in ['3632307990694238', '3632313749473288', '3632309148322699']:
        max_success_comment_count = 10
    comment_list = hudong_info.get('comment_list', [])
    comment_used_list = hudong_info.get('comment_used', [])
    comment_used_list.extend(exist_comment_text)

    print(f"[{bvid}] [步骤 5/8] 调用 pick_commenters 选择评论者...")
    comment_commenters = pick_commenters(commenter_map, COMMENTER_USAFE_FILE,
                                         n=max_success_comment_count)
    print(f"[{bvid}] [步骤 5/8] pick_commenters 调用完成，选择了 {len(comment_commenters)} 个评论者。")

    print(f"[{bvid}] [步骤 6/8] 准备调用 post_comments_once 发送评论...")
    post_comments_once(
        commenter_list=comment_commenters,
        comment_list=comment_list,
        bvid=bvid,
        max_success_comment_count=max_success_comment_count,
        comment_used_list=comment_used_list,
        path_exists=path_exists,
        max_workers=5,
        jitter=(0.4, 1.0)
    )
    print(f"[{bvid}] [步骤 6/8] post_comments_once 调用完成。")

    hudong_info['comment_used'] = comment_used_list
    if hudong_info.get('last_processed_date') == today:
        last_count = int(hudong_info.get('last_processed_date_count', 0) or 0)
        hudong_info['last_processed_date_count'] = last_count + 1
    else:
        hudong_info['last_processed_date_count'] = 1
    hudong_info['last_processed_date'] = today

    print(f"[{bvid}] [步骤 7/8] 准备等待主人弹幕线程...")
    if danmaku_thread and danmaku_thread.is_alive():
        danmaku_thread.join(timeout=THREAD_JOIN_TIMEOUT)
        if danmaku_thread.is_alive():
            print(f"[{bvid}] 警告：主人弹幕线程在 {THREAD_JOIN_TIMEOUT} 秒后仍未结束。")
        else:
            print(f"[{bvid}] 主人弹幕线程已成功执行完毕。")
    else:
        print(f"[{bvid}] 主人弹幕任务未启动或已执行完毕。")
    print(f"[{bvid}] [步骤 7/8] 主人弹幕线程等待完成。")

    hudong_info['owner_danmu_used'] = owner_danmu_used_list

    print(f"[{bvid}] [步骤 8/8] 准备等待其他用户弹幕线程...")
    t.join(timeout=THREAD_JOIN_TIMEOUT)
    if t.is_alive():
        print(f"[{bvid}] 警告：其他用户弹幕线程在 {THREAD_JOIN_TIMEOUT} 秒后仍未结束。")
        stop_event.set()
    else:
        print(f"[{bvid}] 其他用户弹幕线程已成功执行完毕。")
    print(f"[{bvid}] [步骤 8/8] 其他用户弹幕线程等待完成。")

    hudong_info['danmu_used'] = result.sent_texts

    if watch_thread:
        print(f"[{bvid}] 准备等待 watch_video 后台线程...")
        if watch_thread.is_alive():
            watch_thread.join(timeout=THREAD_JOIN_TIMEOUT)
            if watch_thread.is_alive():
                print(f"[{bvid}] 警告：watch_video 线程在 {THREAD_JOIN_TIMEOUT} 秒后仍未结束。")
            else:
                print(f"[{bvid}] watch_video 线程已成功执行完毕。")
        else:
            print(f"[{bvid}] watch_video 线程此前已自动完成。")

    print(f"[{bvid}] --- process_single_video 结束 ---")
    return hudong_info, False


def statistic_good_video(tasks):


    exist_block_video_info = read_json(BLOCK_VIDEO_BVID_FILE)
    all_bvid_list = list(exist_block_video_info.keys())
    query_4 = {
        "bvid": {
            "$in": all_bvid_list
        }
    }
    blocked_task_list = manager.find_by_custom_query(manager.tasks_collection, query_4)

    block_video_id_str_list = []
    for blocked_task_info in blocked_task_list:
        video_id_list = blocked_task_info.get('video_id_list', [])
        video_id_key = ','.join(sorted(video_id_list))
        block_video_id_str_list.append(video_id_key)
    new_tasks = []

    skip_count = 0
    for task_info in tasks:
        video_id_list = task_info.get('video_id_list', [])
        video_id_key = ','.join(sorted(video_id_list))
        if video_id_key in block_video_id_str_list:
            skip_count += 1
            # print(f"任务 {video_id_list} 在黑名单中，跳过统计。")
            continue
        new_tasks.append(task_info.copy())
    print(f"统计播放量信息，共 {len(new_tasks)} 个任务需要统计。总共任务数: {len(tasks)} 黑名单任务数:{len(block_video_id_str_list)} 跳过任务数: {skip_count}")

    filter_tasks = []
    user_task_info = {}
    user_config = read_json(r'W:\project\python_project\auto_video\config\user_config.json')
    self_user_list = user_config.get('self_user_list', [])

    same_video_tasks_info = {}
    for task_info in new_tasks:
        play_comment_info_list = task_info.get('play_comment_info_list', [])
        create_time = task_info.get('created')
        if create_time:
            if len(play_comment_info_list) < 5 and (time.time() - create_time) > 4 * 3600:
                # print(f"任务 {task_info.get('video_id_list', [])} 播放量数据不足，跳过统计。")
                continue
            play_count_info = get_simple_play_distribution(play_comment_info_list, create_time, interval_minutes=60, max_elapsed_minutes=1500)
            if play_count_info:
                user_name = task_info.get('userName', '未知用户')
                if user_name not in user_task_info:
                    user_task_info[user_name] = []
                user_task_info[user_name].append(task_info)
                task_info['play_count_info'] = play_count_info
                filter_tasks.append(task_info)
                video_id_list = task_info.get('video_id_list', [])
                video_id_key = ','.join(sorted(video_id_list))
                if video_id_key not in same_video_tasks_info:
                    same_video_tasks_info[video_id_key] = []
                same_video_tasks_info[video_id_key].append(task_info)

    for user_name, task_list in user_task_info.items():
        play_count_info_list = [task_info.get('play_count_info', {}) for task_info in task_list]
        averages_info = calculate_averages(play_count_info_list)
        print(f"用户 {user_name} 的平均播放量信息: {averages_info}")
        for task_info in task_list:
            play_count_info = task_info.get('play_count_info', {})
            total_ratio_value = 0
            total_play_count = 0
            valid_count = 0
            copy_play_count_info = play_count_info.copy()
            for key, play_count in play_count_info.items():
                total_play_count = play_count
                average_value = averages_info.get(key, 0)
                if average_value > 0:
                    ratio_key = f"{key}历史比例"
                    ratio_value = play_count / average_value
                    copy_play_count_info[ratio_key] = round(ratio_value, 2)
                    total_ratio_value += ratio_value
                    valid_count += 1
            avg_total_ratio_value = total_ratio_value / valid_count if valid_count > 0 else 0
            copy_play_count_info['平均历史比例'] = round(avg_total_ratio_value, 2)
            copy_play_count_info['平均播放量'] = round(total_play_count * 60 / int(key) if int(key) > 0 else 0, 2)
            task_info['play_count_info'] = copy_play_count_info


    time_diff_list = [6, 12, 24, 36, 48, 60, 72,84,96,108,120,132, 144]
    final_good_task_list = []
    for time_diff in time_diff_list:
        same_video_tasks_info = {}
        video_type_tasks_info = {}
        all_tasks_list = []
        # 获取当前时间戳往前移动 time_diff 小时的时间戳
        pre_timestamp = int(time.time()) - time_diff * 3600

        for task_info in new_tasks:
            create_time = task_info.get('created')
            if create_time:
                if pre_timestamp > create_time:
                    continue
                play_comment_info_list = task_info.get('play_comment_info_list', [])
                if len(play_comment_info_list) < 5 and (time.time() - create_time) > 4 * 3600:
                    # print(f"任务 {task_info.get('video_id_list', [])} 播放量数据不足，跳过统计。")
                    continue
                all_tasks_list.append(task_info)
                video_id_list = task_info.get('video_id_list', [])
                video_id_key = ','.join(sorted(video_id_list))
                if video_id_key not in same_video_tasks_info:
                    same_video_tasks_info[video_id_key] = []
                same_video_tasks_info[video_id_key].append(task_info)



        real_good_video_list = []
        for video_id_key, task_list in same_video_tasks_info.items():
            aggregated_info = {}
            for task_info in task_list:
                play_count_info = task_info.get('play_count_info', {})
                for key, value in play_count_info.items():
                    if value is None: continue
                    aggregated_info.setdefault(key, []).append(value)

            average_info = {}
            for k, v in aggregated_info.items():
                if len(v) == len(task_list) or (len(v) > 2 and (len(task_list) - len(v)) / len(task_list) < 0.3):
                    log_values = [math.log1p(math.log1p(x)) for x in v]
                    avg_log = sum(log_values) / len(log_values)
                    # 还原：exp(exp(y)-1)-1
                    average_info[k] = math.expm1(math.expm1(avg_log))
            if average_info.get('平均历史比例') < 1:
                continue
            for task_info in task_list:
                task_info['average_info'] = average_info
                task_info['same_count'] = len(task_list)
                task_info['aggregated_info'] = aggregated_info
                task_info['final_score'] = task_info['average_info'].get('平均历史比例') * task_info['same_count'] * task_info['average_info'].get('平均播放量')
                if task_info['same_count'] < 2:
                    task_info['final_score'] *= 0.5
            # if task_info['final_score'] < 200:
            #     continue

            # 2. 按平均历史比例从高到低排序
            temp_task_list = sorted(task_list, key=lambda x: x.get('play_count_info', {}).get('平均历史比例', 0),reverse=True)
            if temp_task_list:
                # 3. 将全场最高比例的任务加入（不管是哪位用户的）
                top_task = temp_task_list[0]
                real_good_video_list.append(top_task)
                if top_task.get('userName') in self_user_list:
                    best_external_task = next((t for t in temp_task_list if t.get('userName') not in self_user_list),None)
                    if best_external_task:
                        real_good_video_list.append(best_external_task)
                        # print(f"在用户 {top_task.get('userName')} 的{top_task.get('upload_params').get('title')}视频中，找到了自有账号，加入了外部账号用户 {best_external_task.get('userName')} {best_external_task.get('upload_params').get('title')} 的视频。")
        video_tag_tasks_info = {}
        for task_info in real_good_video_list:
            upload_info_list = task_info.get('upload_info', [])
            video_type, tags_info = gen_true_type_and_tags(upload_info_list)
            for tag, count in tags_info.items():
                if tag not in video_tag_tasks_info:
                    video_tag_tasks_info[tag] = []
                video_tag_tasks_info[tag].append(task_info)

            user_name = task_info.get('userName', '未知用户')
            user_type = get_user_type(user_name)
            if user_type not in video_type_tasks_info:
                video_type_tasks_info[user_type] = []
            video_type_tasks_info[user_type].append(task_info)


        print(f"在过去 {time_diff} 小时内，共发现 {len(real_good_video_list)} 个素材的视频。满足时间要求的任务数量为{len(all_tasks_list)}")
        tag_need_count = 1
        for tag, task_list in video_tag_tasks_info.items():
            temp_task_list = sorted(task_list, key=lambda x: (x.get('final_score', 0)), reverse=True)
            for i, task_info in enumerate(temp_task_list[:tag_need_count]):
                if 'choose_reason' not in task_info:
                    task_info['choose_reason'] = []
                final_good_task_list.append(task_info)
                task_info['choose_reason'].append(f'标签 {tag} {time_diff}  排行第{i + 1}名的前5个视频')


        need_count = 10
        for user_type, task_list in video_type_tasks_info.items():
            temp_task_list = sorted(task_list, key=lambda x: (x.get('final_score', 0)), reverse=True)
            for i, task_info in enumerate(temp_task_list[:need_count]):
                if 'choose_reason' not in task_info:
                    task_info['choose_reason'] = []
                final_good_task_list.append(task_info)
                task_info['choose_reason'].append(f'{user_type} {time_diff}  排行第{i + 1}名的前5个视频')

        sorted_tasks = sorted(real_good_video_list, key=lambda x: (x.get('final_score', 0)), reverse=True)
        for i, task_info in enumerate(sorted_tasks[:need_count]):
            if 'choose_reason' not in task_info:
                task_info['choose_reason'] = []
            final_good_task_list.append(task_info)
            task_info['choose_reason'].append(f'全局 {time_diff} 排行第{i+1}名的前5个视频')

    # final_good_task_list = [task_info for task_info in final_good_task_list if task_info.get('final_score', 0) > 50]
    # 对final_good_task_list进行去重，按照task_info的bvid去重
    unique_bvids = set()
    unique_final_good_task_list = []
    # user_config = read_json(r'W:\project\python_project\auto_video\config\user_config.json')
    # self_user_list = user_config.get('self_user_list', [])
    #
    # # 过滤掉final_good_task_list中userName在self_user_list中的任务
    # final_good_task_list = [task_info for task_info in final_good_task_list if task_info.get('userName', '') not in self_user_list]



    for task_info in final_good_task_list:
        user_name = task_info.get('userName', '未知用户')
        is_self = False
        if user_name in self_user_list:
            is_self = True

        video_id_list = task_info.get('video_id_list', [])
        video_id_key = '_'.join(sorted(video_id_list))
        video_id_key = f"{video_id_key}_{'self' if is_self else 'external'}"
        if video_id_key and video_id_key not in unique_bvids:
            unique_bvids.add(video_id_key)
            unique_final_good_task_list.append(task_info)
    # 最终按照choose_reason的长度降序排序
    unique_final_good_task_list.sort(
        key=lambda x: (
            len(x.get('choose_reason', [])),
            x.get('final_score', 0)
        ),
        reverse=True
    )
    user_type_count_info = {}
    for video_info in unique_final_good_task_list:
        user_name = video_info.get('userName', 'dahao')
        user_type = get_user_type(user_name)
        if user_type not in user_type_count_info:
            user_type_count_info[user_type] = 0
        user_type_count_info[user_type] += 1

    print(f"最终筛选出 {len(unique_final_good_task_list)} 个优质视频 {user_type_count_info} 其中有 {len([task_info for task_info in unique_final_good_task_list if task_info.get('same_count', 0) > 1])} 个是不止一个任务。去重前数量: {len(final_good_task_list)} 总共视频数量: {len(filter_tasks)}")

    # 过滤出unique_final_good_task_list中same_count大于1的任务
    # filter_task_list = [task_info for task_info in unique_final_good_task_list if task_info.get('same_count', 0) > 1]
    # unique_final_good_task_list = sorted(unique_final_good_task_list, key=lambda x: (x.get('final_score', 0)), reverse=True)
    # 过滤出unique_final_good_task_list中final_score不存在的任务
    # unique_final_good_task_list = [task_info for task_info in unique_final_good_task_list if task_info.get('final_score', 0) > 0]
    good_tags_info = {}
    # 统计热门的tags
    for task_info in unique_final_good_task_list:
        upload_info_list = task_info.get('upload_info', [])
        video_type, tags_info = gen_true_type_and_tags(upload_info_list)
        user_name = task_info.get('userName', 'dahao')
        video_type = get_user_type(user_name)
        task_info['video_type'] = video_type
        choose_reason = task_info.get('choose_reason', [])
        choose_reason_len = len(choose_reason)
        if video_type:
            if video_type not in good_tags_info:
                good_tags_info[video_type] = {}
            for tag, count in tags_info.items():
                good_tags_info[video_type][tag] = good_tags_info[video_type].get(tag, 0) + count * choose_reason_len
    temp_info = {}
    # 获取当前的时间，以字符串形式打出来
    current_time_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    temp_info['current_time_str'] = current_time_str
    temp_info['good_tags_info'] = good_tags_info
    temp_info['good_video_list'] = unique_final_good_task_list
    print(f"保存优质视频统计信息，时间点: {current_time_str} 共 {len(unique_final_good_task_list)} 个优质视频。")
    save_json(STATISTIC_PLAY_COUNT_FILE, temp_info)


def update_block_video(config_map):
    """
    更新被删除的数据信息
    :param config_map:
    :return:
    """

    exist_block_video_info = read_json(BLOCK_VIDEO_BVID_FILE)
    for uid, target_value in config_map.items():
        try:
            arc_video_info = get_bilibili_archives(target_value['total_cookie'])
            arc_video_list = arc_video_info.get('data', {}).get('arc_audits', [])
            for arc_video_info in arc_video_list:
                archive_info = arc_video_info.get('Archive', {})
                bvid = archive_info.get('bvid', '')
                exist_block_video_info[bvid] = arc_video_info
            print(f"获取用户 {uid} 的视频列表完成，共 {len(arc_video_list)} 个视频。")
            save_json(BLOCK_VIDEO_BVID_FILE, exist_block_video_info)
        except Exception as e:
            print(f"获取用户 {uid} 的视频列表失败: {e}")
            continue





def fun(manager):
    global NEED_UPDATE_SIGN
    try:
        now = datetime.now()
        today_midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        pre_midnight = today_midnight - timedelta(days=5)

        recent_uploaded_tasks = manager.find_tasks_after_time_with_status(pre_midnight, [TaskStatus.UPLOADED])


        processed_count = 0
        print("开始执行 fun 函数...当前时间:", datetime.now().isoformat())
        stop_event.clear()
        config_map = init_config()
        commenter = BilibiliCommenter(total_cookie=total_cookie, csrf_token=csrf_token)
        commenter_map = {}
        for key, detail_config in config_map.items():
            name = detail_config.get('name', key)
            if name in ['xiaoxiaosu']:
                continue
            cookie = detail_config.get('total_cookie', '')
            all_params = detail_config.get('all_params', {})
            commenter_map[key] = BilibiliCommenter(
                total_cookie=cookie,
                csrf_token=detail_config.get('BILI_JCT', ''),
                all_params=all_params,
            )
            print(f"已创建评论者 {name} (UID: {key})")
        print(f"共创建 {len(commenter_map)} 个评论者实例。")

        bvid_file_data = read_json(USER_BVID_FILE)
        all_bvid_file_data = read_json(ALL_BVID_FILE)

        update_block_video(config_map)
        bvid_uid_map = {}
        all_found_videos = []
        for uid in config_map.keys():
            name = config_map[uid].get('name', uid)
            # if name != 'mama':
            #     continue

            if NEED_UPDATE_SIGN:
                detail_config = config_map[uid]
                name = detail_config.get('name', key)

                signature = random.choice(signatures)
                cookie = detail_config.get('total_cookie', '')
                result = update_bili_user_sign(cookie, signature)
                print(f"更新用户签名结果: {result} {name}")

            logging.info(f"  > 正在获取UP主(UID: {uid} {name})的最新动态...")
            temp_found_videos = commenter.get_user_videos(mid=uid, desired_count=25)
            bvid_uid_map.update({video.get('bvid'): uid for video in temp_found_videos if 'bvid' in video})
            all_found_videos.extend(temp_found_videos)
            bvid_file_data[name] = temp_found_videos
            for video in temp_found_videos:
                all_bvid_file_data[video.get('bvid')] = video
            save_json(ALL_BVID_FILE, all_bvid_file_data)
            save_json(USER_BVID_FILE, bvid_file_data)

        need_update_task_list, recent_uploaded_tasks = update_play_count(recent_uploaded_tasks, bvid_file_data)
        manager.upsert_tasks(need_update_task_list)
        statistic_good_video(recent_uploaded_tasks)



        filter_tasks = []
        three_hours_ago = datetime.now() - timedelta(hours=20)

        for task_info in need_update_task_list:
            uploaded_time = task_info.get('uploaded_time')
            hudong_info = task_info.get('hudong_info', {})
            last_processed_date = hudong_info.get('last_processed_date', '')
            if last_processed_date:
                continue
            if uploaded_time and uploaded_time >= three_hours_ago:
                filter_tasks.append(task_info)
        print(f"共找到 {len(filter_tasks)} 个最近3小时内上传的任务。")
        count = 0

        for task_info in filter_tasks:
            start_time = time.time()
            count += 1
            bvid = task_info.get('upload_result', {}).get('bvid', '')
            uid = bvid_uid_map.get(bvid, '未知UID')
            hudong_info, is_skip = process_single_hudong_task(task_info, commenter_map, uid)
            if not is_skip:
                task_info['hudong_time'] = datetime.now()
                task_info['hudong_info'] = hudong_info
                manager.upsert_tasks([task_info])
                processed_count += 1
            print(
                f"视频 {bvid} 的互动信息已生成并保存。耗时: {time.time() - start_time:.2f} 秒 进度: {count}/{len(filter_tasks)} {datetime.now().isoformat()}")
        NEED_UPDATE_SIGN = False
        print(
            f"所有视频处理所有完成所有，正在保存数据..当前时间: {datetime.now().isoformat()} 共处理 {processed_count} 个视频。共找到 {len(recent_uploaded_tasks)} 个视频")
    except Exception as e:
        traceback.print_exc()
    finally:
        stop_event.set()


if __name__ == '__main__':
    # config_map = init_config()
    # mid_list = config_map.keys()
    # block_all_author(mid_list, action_type=6)

    # 更好的统计出好视频或者说是好的素材也就是说每次都爆的才证明是好视频

    mongo_base_instance = gen_db_object()
    manager = MongoManager(mongo_base_instance)
    # 启动定时任务线程
    threading.Thread(target=run_periodically, args=(manager,), daemon=True).start()

    # 主线程可用于其他任务，或者继续保持程序运行
    while True:
        time.sleep(10)
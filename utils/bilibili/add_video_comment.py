import datetime
import os
import random
import time
import traceback
from typing import Any, Dict, List, Optional
from multiprocessing import Pool

from utils.auto_web.gemini_auto import generate_gemini_content_playwright
from utils.bilibili.comment import BilibiliCommenter
from utils.bilibili.get_comment import get_bilibili_comments
from utils.common_utils import read_json, init_config, save_json, string_to_object, read_file_to_str
from utils.gemini import get_llm_content
from utils.mongo_base import gen_db_object
from utils.mongo_manager import MongoManager

BASE_DIR = r'W:\project\python_project\auto_video\utils\temp'

RECORD_EXPIRATION_DAYS = 2
DEFAULT_MAX_REPLIES_PER_RUN = 1
REDUCED_MAX_REPLIES_PER_RUN = 1
EXISTING_REPLIES_THRESHOLD = 1
RETRY_INTERVAL_SECONDS = 10


def load_all_replay_info():
    local_replay_info_file = os.path.join(BASE_DIR, 'formatted_video_data.json')
    return read_json(local_replay_info_file)


def _truncate_field(value, limit):
    """把 value 转为字符串，去首尾空白并把换行替换为空格，然后截取前 limit 个字符。"""
    if value is None:
        return ''
    s = str(value).strip().replace('\n', ' ')
    return s[:limit]


def gen_final_property_replay(video_info, all_replay_info):
    """根据视频信息生成合适的商品信息"""
    pure_all_replay_info = [{
        '名称': _truncate_field(item.get('名称'), 10),
        '剧情简介': _truncate_field(item.get('剧情简介'), 100),
        '演员': _truncate_field(item.get('演员'), 10),
        '题材': _truncate_field(item.get('题材'), 10)
    } for item in all_replay_info]

    print(f"正在生成最终商品信息，视频信息")
    retry_delay = 10
    max_retries = 3

    draft_video_script_info = video_info.get('draft_video_script_info', [{}])[0]
    comment_list = video_info.get('hudong_info', {}).get('comment_list', [])

    # 按照c[1]降序排序，截取前100
    temp_comments = sorted([(c[0], c[1]) for c in comment_list], key=lambda x: x[1], reverse=True)[:100]

    format_video_info = {
        'titles': video_info.get('upload_params', {}).get('title', ''),
        'video_summary': draft_video_script_info.get('one_sentence_summary', ''),
        'scene_summary_list': [scene['scene_summary'] for scene in
                               draft_video_script_info.get('scene_sourcing_plan', [])],
        'comments': temp_comments,
        'videos': pure_all_replay_info
    }

    PROMPT_FILE_PATH = r'W:\project\python_project\auto_video\application\prompt\推荐视频b站好片.txt'
    prompt = f"{read_file_to_str(PROMPT_FILE_PATH)}\n输入信息如下:\n{format_video_info}"
    model_name_list = ["gemini-3-flash-preview", "gemini-2.5-flash", "gemini-flash-lite-latest"]
    raw = ""
    for attempt in range(1, max_retries + 1):

        model_name = random.choice(model_name_list)
        try:
            # raw = get_llm_content(prompt=prompt, model_name=model_name)
            gen_error_info, raw = generate_gemini_content_playwright(prompt, file_path=None,
                                                                              model_name="gemini-3-flash-preview")

            return string_to_object(raw)
        except Exception as e:
            print(f"[ERROR] 生成视频信息失败 (尝试 {attempt}/{max_retries}): {e} {raw}")
            if attempt < max_retries:
                print(f"[INFO] 正在重试... (等待 {retry_delay} 秒)")
                time.sleep(retry_delay)
            else:
                print("[ERROR] 达到最大重试次数，失败.")
                return None
            traceback.print_exc()


def _is_rpid_in_comments(rpid: int, comments: List[Dict[str, Any]]) -> bool:
    """健壮地检查 rpid 是否存在于评论列表中。"""
    if not comments:
        return True
    return str(rpid) in str(comments)


def _process_single_video(bvid: str, record: Dict[str, Any], commenter_pool) -> Dict[str, Any]:
    """处理单个视频的回复逻辑，返回更新后的记录。"""
    updated_record = record.copy()

    rpid = updated_record.get('rpid')
    if not rpid:
        print(f"视频 {bvid} 缺少 rpid，无法处理。")
        return updated_record

    comments = get_bilibili_comments(bvid)
    if not _is_rpid_in_comments(rpid, comments):
        print(f"视频 {bvid} 的目标评论 {rpid} 不存在或已删除，标记为删除。")
        updated_record['status'] = 'delete'
        return updated_record

    exist_shill_comments = updated_record.get('exist_shill_comments', [])
    all_shill_comments = updated_record.get('shill_comments', [])
    comments_to_send = [c for c in all_shill_comments if c not in exist_shill_comments]

    if not comments_to_send:
        print(f"视频 {bvid} 没有新的评论文案可以发送。")
        return updated_record

    max_replies_this_run = (REDUCED_MAX_REPLIES_PER_RUN
                            if len(exist_shill_comments) >= EXISTING_REPLIES_THRESHOLD
                            else DEFAULT_MAX_REPLIES_PER_RUN)

    success_count = 0
    exist_shill_users = updated_record.get('exist_shill_users', [])

    for shill_comment in comments_to_send:
        if success_count >= max_replies_this_run:
            print(f"已达到本次运行回复上限 ({max_replies_this_run})，停止回复视频 {bvid}。")
            break

        available_commenters = [c for c in commenter_pool if c[0] not in exist_shill_users]
        random.shuffle(available_commenters)

        for commenter_name, commenter in available_commenters:
            reply_rpid, reason = commenter.reply_to_comment(
                bvid=bvid,
                message_content=shill_comment,
                root_rpid=rpid,
                parent_rpid=rpid
            )

            if reply_rpid:
                success_count += 1
                exist_shill_comments.append(shill_comment)
                exist_shill_users.append(commenter_name)
                print(f"✅ {commenter_name} 回复成功: 视频 {bvid}, 内容: {shill_comment[:30]}...")
                break
            else:
                print(f"❌ {commenter_name} 回复失败: {reason}")
                time.sleep(RETRY_INTERVAL_SECONDS)
                if '无法获取有效的视频信息' in reason or '删除' in reason:
                    print(f"视频 {bvid} 或评论似乎已失效，标记为删除。")
                    updated_record['status'] = 'delete'
                    return updated_record
                time.sleep(RETRY_INTERVAL_SECONDS)

    updated_record['exist_shill_comments'] = exist_shill_comments
    updated_record['exist_shill_users'] = exist_shill_users
    return updated_record


def _initialize_commenters(config_map: Dict[str, Any], user_to_exclude: str) -> Dict[str, 'BilibiliCommenter']:
    """根据配置初始化所有评论员实例，并排除指定用户。"""
    commenter_map = {}
    for uid, config in config_map.items():
        name = config.get('name', uid)
        if name == user_to_exclude:
            continue

        try:
            commenter = BilibiliCommenter(
                total_cookie=config.get('total_cookie', ''),
                csrf_token=config.get('BILI_JCT', ''),
                all_params=config.get('all_params', {})
            )
            commenter_map[name] = commenter
            print(f"已创建评论者 {name} (UID: {uid})")
        except Exception as e:
            print(f"创建评论者 {name} (UID: {uid}) 失败: {e}")

    return commenter_map


def _should_skip_video(record: Dict[str, Any], bvid: str, today: str) -> Optional[str]:
    """检查是否应跳过当前视频，如果需要则返回原因，否则返回 None。"""
    send_time = record.get('send_time', 0)
    if time.time() - send_time > RECORD_EXPIRATION_DAYS * 86400:
        return f"记录 {bvid} 已超过 {RECORD_EXPIRATION_DAYS} 天，跳过。"

    if len(record.get('exist_shill_comments', [])) >= EXISTING_REPLIES_THRESHOLD:
        if not record.get('rpid') or not record.get('good_name'):
            return f"记录 {bvid} 已有足够回复但信息不全，跳过。"
        if record.get('last_processed_date') == today:
            return f"记录 {bvid} 今日已处理，跳过。"

    return None


def auto_replay_refactored(user_name: str):
    """自动扫描并回复指定用户的置顶评论，以增加商品购买几率。"""
    print(f"\n🚀 开始为用户 {user_name} 的视频增加置顶文案回复...当前时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        all_records_file = f"{BASE_DIR}/{user_name}_replay_video_info.json"
        all_records = read_json(all_records_file)
        seven_days_ago = time.time() - 1 * 24 * 60 * 60
        all_records = {k: v for k, v in all_records.items() if v.get('send_time', 0) >= seven_days_ago}
        config_map = init_config()
    except FileNotFoundError:
        print(f"错误：找不到文件 {all_records_file} 或配置文件。")
        return
    except Exception as e:
        print(f"加载数据时出错: {e}")
        return

    commenters = _initialize_commenters(config_map, user_to_exclude=user_name)
    if not commenters:
        print("没有可用的评论员账号，程序终止。")
        return
    commenter_pool = list(commenters.items())

    today = datetime.date.today().isoformat()
    processed_count = 0
    total_records = len(all_records)

    for bvid, record in all_records.items():
        processed_count += 1
        print(f"\n[{processed_count}/{total_records}] 正在处理视频 BVID: {bvid}...")

        skip_reason = _should_skip_video(record, bvid, today)
        if skip_reason:
            print(f"⏭️  跳过: {skip_reason}")
            record['last_processed_date'] = today
            if "已达上限" not in skip_reason:
                record['process_count'] = record.get('process_count', 0) + 1
            all_records[bvid] = record
            continue

        try:
            updated_record = _process_single_video(bvid, record, commenter_pool)
            updated_record['last_processed_date'] = today
            updated_record['process_count'] = updated_record.get('process_count', 0) + 1
            all_records[bvid] = updated_record
            save_json(all_records_file, all_records)

        except Exception as e:
            print(f"处理视频 {bvid} 时发生未知严重错误: {e}")
            traceback.print_exc()
            record['last_processed_date'] = today
            record['process_count'] = record.get('process_count', 0) + 1
            all_records[bvid] = record
            save_json(all_records_file, all_records)

    try:
        save_json(all_records_file, all_records)
        print(f"\n✅ 全部处理完成，已将更新后的 {len(all_records)} 条记录保存至文件。")
    except Exception as e:
        print(f"最终保存文件时出错: {e}")


def send_replay_comment(commenter: Any, bvid: str, record_info):
    """发送商品评论到指定的 B 站视频，并将评论置顶。"""
    property_goods = record_info['property_goods']
    sorted_recs = record_info.get('final_goods', [])
    print(f"\n\n正在发送回复性评论到视频 {bvid}")
    print(f"找到 {len(sorted_recs)} 条电影推荐。")

    random.shuffle(sorted_recs)
    for rec in sorted_recs:
        title: str = rec.get('名称', '')
        if not title:
            continue

        target_good = next((pg for pg in property_goods if pg.get('名称') == title),
                           property_goods[0] if property_goods else {})
        movie_link = target_good.get('链接', '')
        if not movie_link:
            continue

        pinned_text: str = rec.get('置顶评论', '').strip()
        comment_body = f"{pinned_text}\n{movie_link}"

        print(f"正在发布电影推荐评论: 视频 {bvid}，电影 {title} comment_body: {comment_body}")
        rpid = commenter.post_comment(bvid=bvid, message_content=comment_body)
        if not rpid:
            continue

        if commenter.pin_comment(bvid=bvid, rpid=rpid):
            record_info['comment_body'] = comment_body
            record_info['shill_comments'] = rec.get('互动评论', [])
            print(f"✅ 已成功发送并置电影推荐评论: 视频 {bvid}，电影 {title} comment_body: {comment_body}")
            time.sleep(60)
            return rpid, target_good.get('名称', '')

    print(f"⚠️ 未能发送或置顶任何商品评论到视频 {bvid}")
    return None, None


def add_replay_comment_for_video(task_info_list, user_name='qiqi'):
    """为视频增加合适的商品链接"""
    all_replay_info = load_all_replay_info()
    print(f"\n\n开始为用户 {user_name} 的视频增加视频推荐评论...")
    config_map = init_config()
    all_records_file = f"{BASE_DIR}/{user_name}_replay_video_info.json"

    uid = next((key for key, value in config_map.items() if value.get('name') == user_name), None)
    if not uid:
        print(f"未找到用户 {user_name} 的配置，程序终止。")
        return

    commenter = BilibiliCommenter(
        total_cookie=config_map[uid]['total_cookie'],
        csrf_token=config_map[uid].get('BILI_JCT', ''),
        all_params=config_map[uid].get('all_params', {})
    )

    all_records = read_json(all_records_file)
    success_bvids = []
    for rec in all_records.values():
        bvid = rec.get('bvid')
        if not bvid:
            continue
        if rec.get('status') == 'success' and rec.get('rpid'):
            success_bvids.append(bvid)
            continue
        try:
            if int(rec.get('process_count', 0)) > 1:
                success_bvids.append(bvid)
        except (TypeError, ValueError):
            pass

    processed_bvids = set(success_bvids)
    print(f"已处理 {len(all_records)} 条记录，其中 {len(success_bvids)} 条成功。")

    videos_to_process = [video for video in task_info_list if video.get('bvid') not in processed_bvids][:10]
    print(f"{user_name} 找到 {len(videos_to_process)} 个未处理的视频。总共任务数量：{len(task_info_list)}")

    for video in videos_to_process:
        bvid = video.get('bvid', '')
        if not bvid:
            print(f"视频未找到对应bvid信息，跳过。")
            continue

        try:
            print(f"\n\n正在处理视频 {bvid}，标题: {video.get('upload_params', {}).get('title', '')}...")
            record = all_records.setdefault(bvid, {})
            record.update({'bvid': bvid, 'user_name': user_name})
            save_json(all_records_file, all_records)

            if 'final_goods' in record and record['final_goods']:
                print(f"视频 {bvid} 已经有最终商品信息，跳过LLM生成。")
                final_goods = record['final_goods']
            else:
                final_goods = gen_final_property_replay(video, all_replay_info)
                record['final_goods'] = final_goods
                save_json(all_records_file, all_records)

            if final_goods:
                record['property_goods'] = all_replay_info
                rpid, title = send_replay_comment(commenter, bvid, record)
                if rpid:
                    record.update({
                        'status': 'success',
                        'rpid': rpid,
                        'title': title,
                        'upload_time': time.time(),
                        'send_time': time.time(),
                        'property_goods': []
                    })
                    save_json(all_records_file, all_records)
                # 增加处理的次数
                record['process_count'] = record.get('process_count', 0) + 1
        except Exception as e:
            print(f"处理视频 {bvid} 时出错: {e}")
            traceback.print_exc()
            all_records[bvid]['status'] = 'error'
            all_records[bvid]['error_message'] = str(e)
            save_json(all_records_file, all_records)


def process_user(user, all_task):
    """子进程执行逻辑"""
    try:
        task_info_list = [task for task in all_task if task.get('userName') == user]
        start_time = time.time()
        print(f"[{time.strftime('%X')}] 子进程开始处理用户: {user}")

        add_replay_comment_for_video(task_info_list, user_name=user)
        auto_replay_refactored(user)

        print(f"[{time.strftime('%X')}] 子进程完成用户: {user} 处理，耗时: {time.time() - start_time:.2f} 秒")
    except Exception as e:
        print(f"[{time.strftime('%X')}] 子进程处理用户 {user} 时出错: {e}")
        traceback.print_exc()


def run_once(username_list):
    mongo_base_instance = gen_db_object()
    manager = MongoManager(mongo_base_instance)
    query = {
        "status": "已投稿",
        "create_time": {
            "$gt": datetime.datetime.now() - datetime.timedelta(hours=24 * 1)
        }
    }
    all_task = manager.find_by_custom_query(manager.tasks_collection, query)
    # 获取all_task所有的userName字段，去重后形成一个列表
    all_username_list = list(set(task.get('userName') for task in all_task if task.get('userName')))


    print(f"当前配置的用户列表:{len(username_list)}{len(all_username_list)} 个 {username_list} {all_username_list}")
    processes_count = 2
    print(f"--- {processes_count} 个进程启动，准备以并行进程处理用户 ---")

    with Pool(processes=processes_count) as pool:
        pool.starmap(
            process_user,
            [(user, all_task) for user in all_username_list]
        )

    print("--- 所有用户处理完成 ---")


if __name__ == '__main__':


    while True:
        user_config = read_json(r'W:\project\python_project\auto_video\config\user_config.json')
        username_list = user_config.get('video_recommend_user_list', [])
        start_time = time.time()
        run_once(username_list)

        elapsed_time = time.time() - start_time
        wait_time = max(1800 - elapsed_time, 0)

        print(f"本轮执行时间: {elapsed_time:.2f} 秒，等待 {wait_time:.2f} 秒后开始下一轮...")
        time.sleep(wait_time)
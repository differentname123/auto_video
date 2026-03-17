import datetime
import math
import os
import difflib
import random
import time
import traceback
from typing import Any, Dict, List, Optional
from multiprocessing import Pool

from utils.auto_web.gemini_auto import generate_gemini_content_playwright
from utils.bilibili.comment import BilibiliCommenter
from utils.bilibili.get_comment import get_bilibili_comments
from utils.common_utils import read_json, init_config, save_json, string_to_object, read_file_to_str
# 清理了未使用的 get_llm_content
from utils.gemini_web import generate_gemini_content_managed
from utils.mongo_base import gen_db_object
from utils.mongo_manager import MongoManager
from utils.taobao.search_engine import update_and_search

BASE_DIR = r'W:\project\python_project\auto_video\utils\temp\goods'

RECORD_EXPIRATION_DAYS = 2
DEFAULT_MAX_REPLIES_PER_RUN = 1
REDUCED_MAX_REPLIES_PER_RUN = 1
EXISTING_REPLIES_THRESHOLD = 1
RETRY_INTERVAL_SECONDS = 10


# ==========================================
# 提取的通用 LLM 与数据处理工具函数
# ==========================================

def _extract_video_info_for_llm(video_info: Dict[str, Any]) -> Dict[str, Any]:
    """提取视频信息并格式化，供LLM生成使用"""
    draft_video_script_info = video_info.get('draft_video_script_info', [{}])[0]
    comment_list = video_info.get('hudong_info', {}).get('comment_list', [])

    # 按照c[1]降序排序，截取前100
    temp_comments = sorted(
        [{"content": c[0], "likes": c[1]} for c in comment_list if c[0]],
        key=lambda x: x["likes"],
        reverse=True
    )[:20]

    return {
        'video_title': video_info.get('upload_params', {}).get('title', ''),
        'video_summary': draft_video_script_info.get('one_sentence_summary', ''),
        'video_scenes': [scene.get('scene_summary', '') for scene in
                               draft_video_script_info.get('scene_sourcing_plan', [])],
        'comments': temp_comments,
    }


def _call_gemini_with_retry(prompt_file_path: str, format_video_info: Dict[str, Any]) -> Any:
    """统一的 LLM 调用和重试逻辑"""
    prompt = f"{read_file_to_str(prompt_file_path)}\n{format_video_info}"
    model_name_list = ["gemini-3-flash-preview", "gemini-2.5-flash", "gemini-flash-lite-latest"]

    retry_delay = 10
    max_retries = 3
    raw = ""

    for attempt in range(1, max_retries + 1):
        model_name = random.choice(model_name_list)
        try:
            # gen_error_info, raw = generate_gemini_content_managed(prompt, model_name='gemini-3.0-flash')
            gen_error_info, raw = generate_gemini_content_playwright(prompt, file_path=None,
                                                                              model_name="gemini-3-flash-preview")

            return string_to_object(raw)
        except Exception as e:
            print(f"[ERROR] 调用大模型失败 (尝试 {attempt}/{max_retries}): {e} {raw}")
            if attempt < max_retries:
                print(f"[INFO] 正在重试... (等待 {retry_delay} 秒)")
                time.sleep(retry_delay)
            else:
                print("[ERROR] 达到最大重试次数，失败.")
                traceback.print_exc()
                return None


# ==========================================
# 业务逻辑层
# ==========================================

def gen_property_good(video_info: Dict[str, Any]) -> Any:
    """根据视频信息生成合适的商品信息"""
    print("正在生成初步商品信息，视频信息")
    format_video_info = _extract_video_info_for_llm(video_info)
    prompt_file = r'W:\project\python_project\auto_video\application\prompt\视频商品推荐.txt'
    return _call_gemini_with_retry(prompt_file, format_video_info)


def gen_final_property_good(video_info: Dict[str, Any], goods_info_list: List[Any]) -> Any:
    """根据视频信息和确切的商品信息生成相应的推荐评论"""
    print("正在生成最终商品信息，视频信息 商品数量:", len(goods_info_list))
    format_video_info = _extract_video_info_for_llm(video_info)

    target_key_list = ["category_name", "item_name", "item_id"]  # 你想要保留的字段名列表

    # 使用列表推导式 + 字典推导式
    filtered_goods_list = [
        {key: item[key] for key in target_key_list if key in item}
        for item in goods_info_list
    ]

    format_video_info['goods'] = filtered_goods_list
    prompt_file = r'W:\project\python_project\auto_video\application\prompt\视频商品最终话术生成.txt'
    # 修复Bug: 只返回对象本身，防止下游调用 .get() 崩溃
    return _call_gemini_with_retry(prompt_file, format_video_info), format_video_info


def _is_rpid_in_comments(rpid: int, comments: List[Dict[str, Any]]) -> bool:
    """健壮地检查 rpid 是否存在于评论列表中。"""
    if not comments:
        return True
    return str(rpid) in str(comments)


def extract_shill_comments(data):
    """
    根据 good_name 提取对应的 shill_comments 到最外层
    """
    target_good_name = data.get("good_name")
    if not target_good_name:
        return data

    # 遍历 final_goods 中的 product_recommendations
    final_goods = data.get("final_goods", {}).get("product_recommendations", [])

    for item in final_goods:
        # 使用 goodsName 进行匹配
        if item.get("goodsName") == target_good_name:
            # 抽取 shill_comments
            shill_comments = item.get("shill_comments", [])
            # 将其添加到最外层
            data["shill_comments"] = shill_comments
            break

    return data


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
    if 'shill_comments' not in updated_record:
        updated_record = extract_shill_comments(updated_record)
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


def process_video_replies(user_name: str):
    """自动扫描并回复指定用户的置顶评论，以增加商品购买几率。"""
    print(f"\n🚀 开始为用户 {user_name} 的视频增加置顶文案回复...当前时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        all_records_file = f"{BASE_DIR}/{user_name}_replay_goods_info.json"
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


def filter_property_good(goods_list, limit_count=10):
    if len(goods_list) <= limit_count:
        return sorted(goods_list, key=lambda x: x.get('goods_score', 0), reverse=True)

    def get_id(item):
        return f"{item.get('item_name', '')} {item.get('category_name', '')} {item.get('short_title', '')}"

    sorted_goods = sorted(goods_list, key=lambda x: x.get('goods_score', 0), reverse=True)
    selected_goods = [sorted_goods.pop(0)]

    # 用于存储每次加入时与已选集合的最高相似度
    sim_scores = []

    while len(selected_goods) < limit_count and sorted_goods:
        best_idx = 0
        min_max_sim = 1.0

        for i, item in enumerate(sorted_goods):
            item_id = get_id(item)
            max_sim_with_selected = max(
                difflib.SequenceMatcher(None, item_id, get_id(selected)).ratio()
                for selected in selected_goods
            )

            if max_sim_with_selected < min_max_sim:
                min_max_sim = max_sim_with_selected
                best_idx = i

        # 记录本次被选中项的相似度分值
        sim_scores.append(min_max_sim)
        selected_goods.append(sorted_goods.pop(best_idx))

    # 输出日志
    if sim_scores:
        print(f"最后选择的数据中，各新增项与已选集的最大相似度分值为: {sim_scores}")
        print(f"其中最高相似度分值为: {max(sim_scores)}")

    return sorted(selected_goods, key=lambda x: x.get('goods_score', 0), reverse=True)

def build_comment_body(pinned_text, rec, kouling):
    reason_icons = [("🔥", "🔥"), ("💡", "✨"), ("🏆", "🎯"), ("💎", "💎"), ("✅", "🌟"), ("⏰", "⚡"), ("❤️", "🌹"), ("🛠️", "👍")]
    goods_icons = [("📦", "📦"), ("🛒", "🛒"), ("📌", "📌"), ("🎁", "🎁"), ("💰", "💰"), ("✨", "✨"), ("📍", "📍"), ("🥇", "🥇")]

    reason_pre, reason_post = random.choice(reason_icons)
    goods_pre, goods_post = random.choice(goods_icons)
    taobao = random.choice(['【🍑 宝】', '【🍑 橙色软件】', '【🍑橙色App】'])

    return (
        f"{pinned_text}\n"
        f"{reason_pre} {rec.get('reason', '')} {reason_post}\n"
        f"{goods_pre} {rec.get('goodsName', '')} {goods_post}\n"
        f"{kouling}长按複，制整段内容，然后迲， 👉{taobao}就能直达。"
    )


def send_good_comment(commenter: Any, bvid: str, final_goods_record: Dict[str, Any]):
    """发送商品评论到指定的 B 站视频，并将评论置顶。"""
    print(f"\n\n正在发送商品评论到视频 {bvid}")

    recommendations: List[Dict[str, Any]] = (
        final_goods_record
        .get('final_goods', {})
        .get('product_recommendations', [])
    )

    # 简化排序逻辑，提升可读性
    def _calculate_score(item):
        return float(item.get('estimated_ctr') or 0) * float(item.get('score') or 0)

    sorted_recs = sorted(
        [item for item in recommendations if _calculate_score(item) >= 0],
        key=_calculate_score,
        reverse=True
    )
    print(f"找到 {len(sorted_recs)} 条商品推荐，按预估点击率和评分排序。过滤前推荐数量: {len(recommendations)}")

    property_goods: List[Dict[str, Any]] = final_goods_record.get('property_goods', [])

    for rec in sorted_recs:
        outer_id: str = rec.get('outerId', '')
        if not outer_id:
            continue

        target_good: Optional[Dict[str, Any]] = next(
            (pg for pg in property_goods if pg.get('item_id') == outer_id), None
        )
        if not target_good:
            continue

        taokouling_30d = target_good.get('tpwd_simple', '').strip()
        kouling = taokouling_30d
        # 去除kouling中的 ￥
        if kouling:
            kouling = kouling.replace('￥', '').strip()

        abd_image_path = target_good.get('local_image_path', '')

        if not kouling:
            print(f"⚠️ 商品 {outer_id} 没有有效的短链接，跳过。{taokouling_30d}")
            continue

        pinned_text: str = rec.get('pinned_comment', '').strip()
        comment_body = build_comment_body(pinned_text, rec, kouling)

        print(
            f"正在发布商品评论: 视频 {bvid}，商品 {outer_id} “{rec.get('goodsName', '')}” comment_body: {comment_body}")

        if os.path.exists(abd_image_path):
            rpid = commenter.post_comment(bvid=bvid, message_content=comment_body, image_path=abd_image_path)
        else:
            rpid = commenter.post_comment(bvid=bvid, message_content=comment_body)

        if not rpid:
            continue

        if commenter.pin_comment(bvid=bvid, rpid=rpid):
            final_goods_record['comment_body'] = comment_body
            print(
                f"✅ 已成功发送并置顶商品评论: 视频 {bvid}，商品 {outer_id} “{rec.get('goodsName', '')}” comment_body: {comment_body}")
            return rpid, rec.get('goodsName', ''), rec.get('shill_comments', [])

    print(f"⚠️ 未能发送或置顶任何商品评论到视频 {bvid}")
    return None, None, None


def add_video_goods_comments(task_info_list, user_name='qiqi'):
    """为视频增加合适的商品链接"""
    print(f"\n\n开始为用户 {user_name} 的视频增加商品推荐评论...")
    config_map = init_config()
    all_records_file = f"{BASE_DIR}/{user_name}_replay_goods_info.json"

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

        # 修改点：只有当 final_goods 存在且有值时，才进入后续成功状态的判断
        if rec.get('final_goods'):
            if rec.get('status') == 'success' and rec.get('rpid'):
                success_bvids.append(bvid)
                continue
            try:
                if int(rec.get('process_count', 0)) > 1:
                    success_bvids.append(bvid)
            except (TypeError, ValueError):
                pass

        # if rec.get('status') == 'success' and rec.get('rpid'):
        #     success_bvids.append(bvid)
        #     continue
        # try:
        #     if int(rec.get('process_count', 0)) > 1:
        #         success_bvids.append(bvid)
        # except (TypeError, ValueError):
        #     pass

    processed_bvids = set(success_bvids)
    print(f"已处理 {len(all_records)} 条记录，其中 {len(success_bvids)} 条成功。")

    videos_to_process = [video for video in task_info_list if video.get('bvid') not in processed_bvids][:100]
    print(f"{user_name} 找到 {len(videos_to_process)} 个未处理的视频。总共任务数量：{len(task_info_list)}")

    for video in videos_to_process:
        bvid = video.get('bvid', '')
        if not bvid:
            print("视频未找到对应bvid信息，跳过。")
            continue

        try:
            print(f"\n\n正在处理视频 {bvid}，标题: {video.get('upload_params', {}).get('title', '')}...")
            record = all_records.setdefault(bvid, {})
            record.update({'bvid': bvid, 'user_name': user_name})
            save_json(all_records_file, all_records)

            if 'property_good_info' in record and record['property_good_info']:
                print(f"视频 {bvid} 已经有初步商品信息，跳过LLM生成。")
                property_good_info = record['property_good_info']
            else:
                property_good_info = gen_property_good(video)
                record['property_good_info'] = property_good_info
                save_json(all_records_file, all_records)

            if property_good_info:
                # 修复Bug: 预防LLM漏生成字段导致的 KeyError
                product_recs = property_good_info.get('product_recommendations', [])
                keyword_list = [good.get('product_name', '') for good in product_recs if good.get('product_name')]
                for good in product_recs:
                    keyword_list.extend(good.get('keywords', []))

                keyword_list = list(set(keyword_list))
                limit_count = 50
                top_n = math.ceil(limit_count / len(keyword_list)) if len(keyword_list) > 0 else 5

                property_goods = update_and_search(keyword_list=keyword_list, top_n=top_n)
                print(
                    f"为视频 {bvid} 生成商品信息，关键词列表长度 {len(keyword_list)} 关键词列表：{keyword_list} 每个关键词抓取 {top_n} 个商品")
                property_goods = filter_property_good(property_goods)

                all_records[bvid]['property_goods'] = property_goods
                save_json(all_records_file, all_records)

                if 'final_goods' in record and record['final_goods'] and False:
                    print(f"视频 {bvid} 已经有最终商品信息，跳过。")
                    final_goods = record['final_goods']
                else:
                    final_goods, format_video_info = gen_final_property_good(video, property_goods)
                    all_records[bvid]['final_goods'] = final_goods
                    all_records[bvid]['format_video_info'] = format_video_info
                    save_json(all_records_file, all_records)

                # if final_goods:
                #     rpid, good_name, shill_comments = send_good_comment(commenter, bvid, all_records[bvid])
                #     if rpid:
                #         all_records[bvid]['status'] = 'success'
                #         all_records[bvid]['rpid'] = rpid
                #         all_records[bvid]['good_name'] = good_name
                #         all_records[bvid]['shill_comments'] = shill_comments
                #         all_records[bvid]['upload_time'] = time.time()
                #         all_records[bvid]['send_time'] = time.time()
                #         all_records[bvid]['property_goods'] = []

                # 增加处理的次数
                record['process_count'] = record.get('process_count', 0) + 1
                save_json(all_records_file, all_records)

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

        # 使用了重命名后的函数
        add_video_goods_comments(task_info_list, user_name=user)
        # process_video_replies(user)

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
            "$gt": datetime.datetime.now() - datetime.timedelta(hours=24 * 2)
        }
    }
    all_task = manager.find_by_custom_query(manager.tasks_collection, query)

    print(f"当前配置的用户列表:{len(username_list)}个 {username_list}")
    processes_count = 1
    print(f"--- {processes_count} 个进程启动，准备以并行进程处理用户 ---")
    # username_list = ['ningtao']
    with Pool(processes=processes_count) as pool:
        pool.starmap(
            process_user,
            [(user, all_task) for user in username_list]
        )

    print("--- 所有用户处理完成 ---")


if __name__ == '__main__':

    while True:
        user_config = read_json(r'W:\project\python_project\auto_video\config\user_config.json')
        username_list = user_config.get('allowed_user_list', [])
        start_time = time.time()
        run_once(username_list)

        elapsed_time = time.time() - start_time
        wait_time = max(1800 - elapsed_time, 0)

        print(f"本轮执行时间: {elapsed_time:.2f} 秒，等待 {wait_time:.2f} 秒后开始下一轮...")
        time.sleep(wait_time)
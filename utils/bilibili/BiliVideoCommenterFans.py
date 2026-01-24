#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import requests
import time
import logging
import os
import json
import threading
from queue import Queue, Empty

# 导入自定义工具包 (保持原样)
from utils.bilibili.bili_utils import update_bili_user_sign, modify_relation
from utils.bilibili.comment import BilibiliCommenter
from utils.bilibili.get_comment import get_bilibili_comments
from utils.common_utils import get_config

# ==============================================================================
# 1. 全局配置与常量定义
# ==============================================================================

# 基础API
URL_MODIFY_RELATION = "https://api.bilibili.com/x/relation/modify"

# 文件路径配置
FILES = {
    "DISCOVERED": "DISCOVERED_VIDEOS_FILE.json",
    "PROCESSED_VIDEOS": "processed_bvideos.json",
    "PROCESSED_FIDS": "processed_fids.json",
    "TARGET_FIDS": "target_processed_fids.json"
}

# 运行参数配置
CONFIG = {
    "STRATEGIES": {
        "popular": False,  # 策略1：热门
        "following": False,  # 策略2：关注动态
        "search": True,  # 策略3：搜索
    },
    "TARGET_UIDS": ["443415885", "10330740"],
    "MAX_VIDEOS_PER_SOURCE": 20,
    "REQUEST_TIMEOUT": 10,
    "COOKIE": get_config("dahao_bilibili_total_cookie"),
    "CSRF_TOKEN": get_config("dahao_bilibili_csrf_token"),
}

# 关键词库
KEYWORDS = {
    "TARGET": [
        "互关", "互粉", "互赞", "互助", "新人UP主", "回关", "回粉", "互暖", "互评", "互捞",
        "三连", "求三连", "互三连", "互币", "新人报道", "新人up", "小UP主", "萌新UP", "底层UP主",
        "小透明", "涨粉", "求关注", "求抱团", "抱团取暖", "一起加油", "挑战100粉", "冲击千粉",
        "有粉必回", "有赞必回", "在线秒回", "已关求回"
    ],
    "FOLLOW": [
        "互关", "互粉", "回关", "互赞", "互助", "回粉", "必回", "必回关", "有粉必回",
        "有访必回", "诚信互关", "诚信互粉", "永不取关", "不取关", "赞评必回", "互赞互评",
        "互三连", "互币", "关我必回", "私信秒回", "你关我就关"
    ]
}

# 文案库
TEXTS = {
    "COMMENTS": [
        "如果你喜欢我的内容，不妨关注一下？我也会回关你的！🤝",
        "希望和大家一起进步，关注我，我会回访你的频道。😊",
        "新朋友互关吗？关注我，我也会支持你！",
        "互相关注，共同发展，我期待你的关注和我的回关。",
        "非常乐意和大家互关，关注我，我立刻回粉！",
        "为了更好的交流，我们互相关注吧？我也会去你的频道。👀",
        "欢迎关注我，我也会关注回来的，一起加油！",
        "如果你订阅了我的频道，留言告诉我，我也会去订阅你的！",
        "一起为梦想努力，关注我，我也会回关帮你点赞。",
        "寻找志同道合的朋友互关，关注我，我必回关！",
        "想扩大圈子，关注我，我也会去你的频道留言并关注。",
        "你的关注是对我最大的支持，我也会用关注回报你！",
        "咱们互相支持，你关注我，我也会关注你。✅",
        "小透明求互关，关注我，我秒回！💯",
        "如果你按下关注键，我也会同样按下你的关注键，一起成长！",
        "互关吗朋友？你点关注，我必回访。"
    ],
    "DANMU": [
        "视频质量太高了，已三连！希望我的努力也能被看到~",
        "发现宝藏UP主！果断关注，也希望自己的小作品能被发现。",
        "干货满满，已三连！同为创作者，一起加油！",
        "太用心了，必须支持！我们互相“充电”吧！",
        "制作精良，已点赞关注。也欢迎有空来我这儿坐坐。",
        "这是什么神仙视频！已三连，希望能沾沾大佬的欧气！",
        "学到了很多，感谢UP主！已关注，什么时候我也能做出这种质量啊。",
        "这质量，不点赞关注说不过去。大家一起努力，让好内容发光！",
        "从你的视频里看到了热爱与坚持，已支持！这也激励了我继续创作。",
        "已三连，不解释。同道中人，共勉！",
        "大数据求求了，多推一些这样的优质内容！已关注，也希望我的视频能被推到。",
        "大佬带带我！视频太牛了，默默三连，向您学习！",
        "这才是值得关注的UP主！已三连，希望大家都能为优质内容发电。",
        "关注了，期待UP主更多佳作！也希望我的坚持未来能有回报。",
        "视频做得真好，忍不住三连了。创作者都不容易，一起加油呀！",
        "一键三连！看完感觉自己又充满了创作的动力！",
        "UP主YYDS！已三连，向大佬看齐，努力更新中！",
        "太强了，感觉自己的技能点又提升了！已关注，希望我的分享也能帮到别人。",
        "为你的才华点赞，也为自己的梦想加油。已三连支持！",
        "如此高质量的视频必须三连支持！希望我的主页也能迎来像你一样的观众。"
    ],
    "INTERACTIVE": [
        "弹幕已奉上！从我的【视频页面】关注我吧——无论是谁关注，我都会回关。",
        "点赞评论弹幕都到位，求关注～记得从我的【视频页面】点，我保证一律回关！",
        "支持已送达：已赞已评已发弹幕。到我的【视频页面】关注，我会回关并留言确认。",
        "任务完成：弹幕已发！请从我的【视频页面】关注我，我会马上回关，绝不失约。",
        "已点赞评论弹幕，互帮互助走起来～从我的【视频页面】关注，我一定会回关你。",
        "已经帮忙打气并发弹幕！关注请走【视频页面】，我承诺对每位粉丝回关。",
        "支持全套已完成，求个关注～去我的【视频页面】点关注，我会回关并回访主页。",
        "点赞、评论、弹幕都做好了～从我的【视频页面】关注我，我保证回关并回访你主页。",
        "弹幕已送出！去我的【视频页面】点关注吧，我一定会回关报答支持。",
        "已点赞已评论已弹幕，互助模式开启：请到我的【视频页面】关注，我承诺必回关。",
        "弹幕打卡完成！从我的【视频页面】关注我，我必回关，互相扶持一起成长。",
        "安排好了，弹幕也发了，求个关注～一定要从我的【视频页面】点，我保证回关。",
        "全部支持动作已完成，来我的【视频页面】关注我，我会第一时间回关并留言感谢。",
        "弹幕搞定，拜托去我的【视频页面】点关注，我一定会回关并去你主页看看。",
        "支持已送达！请到我的【视频页面】关注我，我保证回关，一起把账号做大！",
        "点赞评论弹幕齐活～从我的【视频页面】点关注，我一定回关，让我们互相见证成长。",
        "已完成弹幕与互动，诚意满满！请从我的【视频页面】关注，我必回关并回访你主页。",
        "任务打卡：已赞、已评、已发弹幕。记得走我的【视频页面】关注，我会回关不食言。",
        "弹幕已办好～去我的【视频页面】点关注，我承诺对每一位关注者一一回关！",
        "已赞评弹幕齐全，等待关注回馈！请务必从我的【视频页面】点，我一定会回关。"
    ]
}

def read_lines_to_list(file_path: str) -> list:
    """
    读取文件的每一行，返回一个列表，每一行作为一个元素（去掉换行符）
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.rstrip('\n') for line in f]
    return lines

TEXTS['INTERACTIVE'] = read_lines_to_list(r"W:\project\python_project\auto_video\config\comment.json")

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [PID:%(process)d Thread:%(thread)d] - %(levelname)s - %(message)s'
)

# 全局网络会话
SESSION = requests.Session()
SESSION.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Referer': 'https://www.bilibili.com/',
    'Cookie': CONFIG['COOKIE']
})

# 全局状态容器
GLOBAL_STATE = {
    "commenters": [],
    "cookies": [],
    "videos_queue": Queue(),
    "comment_videos_queue": Queue()
}

user_sign_map = {
    'yang': "一枚每天都在和PR死磕的新人女生，希望能做出让你们喜欢的视频。",
    "xue": "积极向上的游戏剪辑少女，你的关注是我的最大动力！",
    "ruruxiao": "从0开始学剪辑的新人女生，希望能得到你的鼓励。",
    "yuhua": "元气少女，正在努力把脑子里叽里咕噜的想法变成好看的视频！",
    "xiaoxiaosu": "梦想是做出很酷的视频，少女正在通往梦想的路上。",
    "junyuan": "一个热爱体育的女生，正在努力学习视频剪辑，用镜头记录热血与感动"

}
# ==============================================================================
# 2. 基础工具函数 (网络与文件IO)
# ==============================================================================

def init_users():
    """初始化多账号信息"""


    user_name_list = ['yang', 'xue', 'ruruxiao', 'yuhua', 'junyuan', 'xiaoxiaosu']
    for name in user_name_list:
        cookie = get_config(f"{name}_bilibili_total_cookie")
        token = get_config(f"{name}_bilibili_csrf_token")

        if not cookie or not token:
            logging.error(f"请在配置文件中设置 {name}_bilibili_total_cookie 和 {name}_bilibili_csrf_token")
            exit(1)

        GLOBAL_STATE["commenters"].append(BilibiliCommenter(cookie, token))
        GLOBAL_STATE["cookies"].append(cookie)
        sign_str = user_sign_map.get(name, "只会回关通过我视频关注我的粉丝，请一定通过我的视频页面来关注我，不然会认为是异常粉丝的")
        res = update_bili_user_sign(cookie, sign_str)
        print(f"签名更新: {res}")


def send_get_request(url, params=None):
    """带重试和延迟的通用GET请求"""
    try:
        time.sleep(random.uniform(1.5, 3.5))
        response = SESSION.get(url, params=params, timeout=CONFIG['REQUEST_TIMEOUT'])
        response.raise_for_status()
        data = response.json()

        if data.get('code', 0) != 0:
            logging.warning(f"API返回错误: code={data.get('code')}, message={data.get('message')}, url={response.url}")
            return None
        return data.get('data')

    except requests.exceptions.RequestException as e:
        logging.error(f"网络请求失败: {e}")
    except json.JSONDecodeError:
        logging.error("无法解析服务器返回的JSON数据")
    return None


def load_json_data(filepath, as_set=False):
    """通用JSON加载，支持返回Set或Dict"""
    if not os.path.exists(filepath):
        return set() if as_set else {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if as_set:
                # 确保集合中的元素转为字符串，防止ID类型不一致
                return {str(item) for item in data}
            return data
    except (json.JSONDecodeError, IOError):
        return set() if as_set else {}


def save_json_data(data, filepath):
    """通用JSON保存"""
    try:
        # 如果是Set，转为List
        if isinstance(data, set):
            data = list(data)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    except IOError as e:
        logging.error(f"保存文件 {filepath} 失败: {e}")


# ==============================================================================
# 3. 视频获取策略模块
# ==============================================================================

def strategy_popular():
    """策略一：获取热门视频"""
    logging.info("执行策略 [Popular]...")
    video_list = []
    url = "https://api.bilibili.com/x/web-interface/popular"
    params = {'ps': CONFIG['MAX_VIDEOS_PER_SOURCE'], 'pn': 1}

    data = send_get_request(url, params)
    if data and 'list' in data:
        for item in data['list']:
            if 'bvid' in item:
                item['_source_strategy'] = 'popular'
                video_list.append(item)
        logging.info(f"  > 热门榜单获取 {len(video_list)} 个视频")
    else:
        logging.warning("  > 热门榜单获取失败")
    return video_list


def strategy_following():
    """策略二：获取关注UP主的动态"""
    logging.info("执行策略 [Following]...")
    if not CONFIG['TARGET_UIDS']:
        logging.warning("  > 未配置目标UID，跳过")
        return []

    video_list = []
    url = "https://api.bilibili.com/x/polymer/web-dynamic/v1/feed/space"

    for uid in CONFIG['TARGET_UIDS']:
        logging.info(f"  > 获取UID: {uid} 动态...")
        params = {'host_mid': uid}
        data = send_get_request(url, params=params)

        if not data or 'items' not in data:
            continue

        found_count = 0
        for item in data['items']:
            # 复杂的动态结构解析
            if item.get('type') != 'DYNAMIC_TYPE_AV':
                continue

            major = item.get('modules', {}).get('module_dynamic', {}).get('major')
            if not major or major.get('type') != 'MAJOR_TYPE_ARCHIVE':
                continue

            video_data = major.get('archive')
            if video_data and 'bvid' in video_data:
                author_info = item.get('modules', {}).get('module_author', {})
                # 补全信息以对齐搜索结果
                video_data['owner'] = {
                    'mid': author_info.get('mid'),
                    'name': author_info.get('name'),
                    'face': author_info.get('face'),
                }
                if 'mid' not in video_data:
                    video_data['mid'] = author_info.get('mid')

                video_data['_source_strategy'] = 'following'
                video_list.append(video_data)
                found_count += 1
                if found_count >= CONFIG['MAX_VIDEOS_PER_SOURCE']:
                    break
        logging.info(f"    - UID {uid} 获取 {found_count} 个新视频")
    return video_list


def strategy_search():
    """策略三：关键词搜索 (包含副作用：重置MAX_VIDEOS_PER_SOURCE)"""
    logging.info("执行策略 [Search]...")
    if not KEYWORDS['TARGET']:
        logging.warning("  > 未配置搜索关键词，跳过")
        return []

    video_list = []
    url = "https://api.bilibili.com/x/web-interface/search/type"
    PAGE_SIZE = 20
    # KEYWORDS['TARGET'] =  KEYWORDS['TARGET'][:1]
    for keyword in KEYWORDS['TARGET']:
        logging.info(f"  > 搜索关键词 '{keyword}'...")
        current_page = 1
        videos_fetched = 0

        while videos_fetched < CONFIG['MAX_VIDEOS_PER_SOURCE']:
            params = {
                'search_type': 'video',
                'keyword': keyword,
                'order': 'pubdate',
                'page': current_page,
                'ps': PAGE_SIZE
            }
            logging.info(f"    - 第 {current_page} 页请求中...")
            data = send_get_request(url, params=params)

            if not data:
                break

            # 兼容 API 结构差异
            search_results = data.get('result', [])
            if not isinstance(search_results, list):
                search_results = data.get('result', {}).get('video', [])

            if not search_results:
                logging.info(f"      - 第 {current_page} 页无数据")
                break

            page_added = 0
            for item in search_results:
                if item.get('type') == 'video' and 'bvid' in item:
                    # 清理标题标签
                    if 'title' in item:
                        item['title'] = item['title'].replace('<em class="keyword">', '').replace('</em>', '')
                    item['_source_strategy'] = 'search'
                    video_list.append(item)
                    videos_fetched += 1
                    page_added += 1

                    if videos_fetched >= CONFIG['MAX_VIDEOS_PER_SOURCE']:
                        break

            logging.info(f"      - 第 {current_page} 页获取 {page_added} 个 (累计: {videos_fetched})")

            if page_added < PAGE_SIZE:
                break  # 到底了

            current_page += 1
            time.sleep(1)

        logging.info(f"  > 关键词 '{keyword}' 结束，共获取 {videos_fetched} 个")
        logging.info("-" * 30)

    # 【副作用警告】: 原始逻辑在此处会重置全局配置
    CONFIG['MAX_VIDEOS_PER_SOURCE'] = 20
    return video_list


def fetch_and_filter_videos():
    """核心调度：获取所有视频 -> 去重 -> 过滤已处理 -> 更新数据库"""
    logging.info("==================== 启动视频获取流程 ====================")

    # 1. 加载历史库
    discovered_map = load_json_data(FILES['DISCOVERED'], as_set=False)
    logging.info(f"加载历史库: {len(discovered_map)} 条")

    # 2. 执行所有策略
    raw_videos = []
    if CONFIG['STRATEGIES']['popular']:
        raw_videos.extend(strategy_popular())
    if CONFIG['STRATEGIES']['following']:
        raw_videos.extend(strategy_following())
    if CONFIG['STRATEGIES']['search']:
        raw_videos.extend(strategy_search())

    # 只保留 raw_videos 中 ”play“ 的值小于10000的视频
    raw_videos = [v for v in raw_videos if v.get('play', 0) < 10000]

    # 3. 本轮内部去重 (保留最新的)
    unique_new_videos = {}
    for vid in reversed(raw_videos):
        if 'bvid' in vid:
            unique_new_videos[vid['bvid']] = vid

    logging.info(f"本轮获取 {len(raw_videos)} 条，去重后 {len(unique_new_videos)} 条")

    # 4. 合并到历史库并保存
    added_count = 0
    for bvid, vid in unique_new_videos.items():
        if bvid not in discovered_map:
            discovered_map[bvid] = vid
            added_count += 1

    if added_count > 0:
        logging.info(f"新增入库 {added_count} 条，更新文件...")
        save_json_data(discovered_map, FILES['DISCOVERED'])
    else:
        logging.info("无新视频入库")

    # 5. 过滤掉“已处理”的视频
    processed_bvid_set = load_json_data(FILES['PROCESSED_VIDEOS'], as_set=True)
    logging.info(f"加载已处理记录: {len(processed_bvid_set)} 条")

    final_todos = [
        v for bvid, v in discovered_map.items()
        if bvid not in processed_bvid_set
    ]
    logging.info(f"最终待处理队列: {len(final_todos)} 条")
    return final_todos


# ==============================================================================
# 4. 工作线程逻辑
# ==============================================================================

def worker_video_fetcher():
    """线程1：定期拉取视频并推送到队列"""
    while True:
        new_videos = fetch_and_filter_videos()

        if new_videos:
            random.shuffle(new_videos)

            # 清空旧队列 (保持原逻辑：强制清空)
            q_vid = GLOBAL_STATE["videos_queue"]
            q_com = GLOBAL_STATE["comment_videos_queue"]

            while not q_vid.empty():
                try:
                    q_vid.get_nowait()
                    q_com.get_nowait()
                except Empty:
                    break

            # 填入新数据
            for v in new_videos:
                q_vid.put(v)
                q_com.put(v)

        logging.info(f"队列更新完毕。当前队列长度: {GLOBAL_STATE['videos_queue'].qsize()}")

        # 长时间休眠
        sleep_sec = random.uniform(1200, 1800)
        logging.info(f"Fetch线程休眠 {int(sleep_sec / 60)} 分钟...")
        time.sleep(sleep_sec)


def process_single_comment_task(video):
    """(辅助) 处理单个视频的评论与弹幕"""
    bvid = video.get('bvid')
    title = video.get('title', '无标题')

    # 检查关键词
    full_text = f"{title} {video.get('description', '')}".lower()
    has_keyword = any(k.lower() in full_text for k in KEYWORDS['FOLLOW'])
    source = video.get('_source_strategy', 'unknown')

    # 逻辑：只有包含关键词才评论，除非是热门视频 (但原代码注释掉should_comment逻辑，此处严格还原)
    # 原代码: should_comment = True (被注释) -> 实际上使用了 if not should_comment and source != 'popular'
    # if not has_keyword and source != 'popular':
    #     logging.info(f"跳过评论: {bvid} (无关键词且非热门)")
    #     return

    logging.info(f"开始评论互动: {bvid} | {title}")

    # 随机打乱评论者
    commenters = GLOBAL_STATE["commenters"]
    random.shuffle(commenters)

    success_count = 0
    max_single_comment_count = 1
    for c in commenters:
        if success_count >= max_single_comment_count:
            logging.info(f"  > 达到单视频评论上限({max_single_comment_count})，停止")
            break

        c_name = getattr(c, "username", str(c))

        try:
            # 1. 发送评论
            txt_comment = random.choice(TEXTS['INTERACTIVE'])
            logging.info(f"  > {c_name} 尝试评论")
            if c.post_comment(bvid, txt_comment, 1, like_video=True):
                logging.info(f"    - 评论成功")
            else:
                logging.error(f"    - 评论失败")

            # 2. 发送弹幕
            txt_danmu = random.choice(TEXTS['DANMU'])
            if c.send_danmaku(bvid, txt_danmu, progress=2000):
                logging.info(f"    - 弹幕成功")
            else:
                logging.error(f"    - 弹幕失败")

            success_count += 1
            time.sleep(random.uniform(1.0, 3.0))

        except Exception as e:
            logging.exception(f"  > {c_name} 操作异常: {e}")

    logging.info(f"视频 {bvid} 互动结束")


def worker_comment_processor():
    """线程2：消费评论队列"""
    # 初始化签名


    queue = GLOBAL_STATE["comment_videos_queue"]
    print("评论处理线程启动")

    while True:
        # 尝试获取视频
        valid_video = None
        start_wait = time.time()
        while time.time() - start_wait < 30:
            try:
                candidate = queue.get(timeout=5)
                if candidate.get('bvid'):
                    valid_video = candidate
                    break
            except Empty:
                logging.warning("评论队列为空")
                break

        if not valid_video:
            time.sleep(random.uniform(5, 10))
            continue

        process_single_comment_task(valid_video)

        # 任务间隔
        time.sleep(random.uniform(100, 110))


def get_users_from_comments(bvid):
    """(辅助) 获取评论区里的潜在互关用户ID"""
    uids = []
    try:
        comments = get_bilibili_comments(bvid)
        for reply in comments:
            msg = reply['content']['message']
            if any(k.lower() in msg for k in KEYWORDS['FOLLOW']):
                uids.append(reply['member']['mid'])
    except Exception as e:
        logging.error(f"获取评论区用户失败: {e}")
    return uids


def worker_follower(csrf_token):
    """线程3：处理关注逻辑"""
    # 加载状态
    processed_videos = load_json_data(FILES['PROCESSED_VIDEOS'], as_set=True)
    processed_fids = load_json_data(FILES['PROCESSED_FIDS'], as_set=True)
    target_processed_fids = load_json_data(FILES['TARGET_FIDS'], as_set=True)

    logging.info(f"关注线程启动，已加载 {len(processed_fids)} 个处理过的用户")

    queue = GLOBAL_STATE["videos_queue"]

    while True:
        try:
            video = queue.get(timeout=30)
            bvid = video.get('bvid', '未知')

            # 标记视频为已处理
            processed_videos.add(bvid)
            save_json_data(processed_videos, FILES['PROCESSED_VIDEOS'])
            logging.info(f"Follow处理视频: {bvid}")

        except Empty:
            continue

        # 提取作者ID
        author_id = video.get('mid')
        if not author_id and 'owner' in video:
            author_id = video['owner'].get('mid')

        if not author_id:
            continue

        # 过滤来源
        source = video.get('_source_strategy', 'unknown')
        # if source != 'popular':
        #     logging.info(f"跳过非热门视频: {source}")
        #     continue

        # -----------------------------------------------------------
        # 核心判定逻辑 (严格保持原代码逻辑)
        # -----------------------------------------------------------
        title = video.get('title', '')
        desc = video.get('description', '')
        # full_text = f"{title} {desc}".lower()
        # 原逻辑：should_follow 实际上被硬编码为 True
        text_to_check = f"{title} {desc}".lower()
        # should_follow = any(keyword.lower() in text_to_check for keyword in KEYWORDS['FOLLOW_KEYWORDS'])
        should_follow = True
        random_trigger = random.random() < 0.1

        if should_follow or random_trigger:
            targets = [author_id]
            # 扩展：从评论区抓人
            # targets.extend(get_users_from_comments(bvid))
            targets = list(set(targets))  # 去重

            author_name = video.get('author') or (video.get('owner', {}).get('name'))
            logging.info(f"命中关注目标! 作者: {author_name} (及评论区共 {len(targets)} 人)")

            for fid in targets:
                fid_str = str(fid)
                if fid_str in processed_fids:
                    logging.info(f"  > 用户 {fid_str} 已处理，跳过")
                    continue

                # 执行关注
                target_processed_fids.add(fid_str)
                processed_fids.add(fid_str)

                for cookie in GLOBAL_STATE["cookies"]:
                    modify_relation(fid_str, 1, cookie)

                # 模拟人为延迟
                time.sleep(random.uniform(40, 60))

            save_json_data(processed_fids, FILES['PROCESSED_FIDS'])
            save_json_data(target_processed_fids, FILES['TARGET_FIDS'])

        else:
            # 即使不关注也记录已处理
            logging.info(f"未命中关注条件，标记作者 {author_id} 已处理")
            processed_fids.add(str(author_id))
            save_json_data(processed_fids, FILES['PROCESSED_FIDS'])

        time.sleep(random.uniform(3, 8))


# ==============================================================================
# 5. 主程序入口
# ==============================================================================

def reset_state_files():
    """重置/删除所有状态文件"""
    for fname in FILES.values():
        if os.path.exists(fname):
            try:
                os.remove(fname)
                logging.info(f"已重置文件: {fname}")
            except OSError as e:
                logging.error(f"删除失败 {fname}: {e}")


def main():
    # 1. 状态重置 (原代码 if True 逻辑)
    if True:
        reset_state_files()

    # 2. 校验配置
    if not CONFIG['COOKIE'] or not CONFIG['CSRF_TOKEN']:
        logging.error("未配置 Cookie 或 CSRF Token，程序退出。")
        return

    logging.info("程序启动...")

    # 3. 初始化资源
    init_users()

    # 4. 启动线程
    t_fetch = threading.Thread(target=worker_video_fetcher, name="VideoFetcher", daemon=True)
    t_fetch.start()

    t_follow = threading.Thread(target=worker_follower, args=(CONFIG['CSRF_TOKEN'],), name="Follower", daemon=True)
    t_follow.start()

    logging.info("评论功能已暂停。如需启用，请取消下方注释。")
    t_comment = threading.Thread(target=worker_comment_processor, name="CommentWorker", daemon=True)
    t_comment.start()

    # 5. 主线程守活
    try:
        while True:
            q_vid_size = GLOBAL_STATE["videos_queue"].qsize()
            q_com_size = GLOBAL_STATE["comment_videos_queue"].qsize()
            logging.info(f"[主线程监控] 待处理视频: {q_vid_size} | 待评论视频: {q_com_size}")
            time.sleep(60)
    except KeyboardInterrupt:
        print("\n程序停止")


if __name__ == '__main__':
    main()
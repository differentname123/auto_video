import functools
import math
import os
import traceback
from datetime import datetime

import requests
import urllib.parse
import time
from hashlib import md5

import requests
import urllib.parse
import time
import random
from hashlib import md5

from application.dig_video import get_need_dig_video_list
from application.video_common_config import ALL_TARGET_TAGS_INFO_FILE, RECENT_HOT_TAGS_FILE
from utils.bilibili.bili_utils import fetch_from_search
from utils.common_utils import read_json, save_json, time_to_ms

# 准备几个常见的 User-Agent
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) Gecko/20100101 Firefox/128.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15"
]

def with_proxy(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
        os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
        try:
            return func(*args, **kwargs)
        finally:
            if 'HTTP_PROXY' in os.environ:
                del os.environ['HTTP_PROXY']
            if 'HTTPS_PROXY' in os.environ:
                del os.environ['HTTPS_PROXY']

    return wrapper


def get_user_videos_public(mid: int, desired_count: int = 30, order: str = 'pubdate', keyword: str = '',
                           proxies: dict = None) -> list:
    """
    独立且免(登录)Cookie的B站用户视频获取函数 - 增强防风控版
    """

    # 使用局部 Session 来自动维持访客状态 (如 buvid3 等跟踪 cookie)
    # 这不需要你输入任何账号信息，仅仅是为了模拟浏览器行为
    session = requests.Session()

    # 随机选择一个 User-Agent
    current_ua = random.choice(USER_AGENTS)
    session.headers.update({
        "User-Agent": current_ua,
        "Referer": f"https://space.bilibili.com/{mid}/video",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    })

    # --- 内部辅助函数：WBI 签名逻辑 ---
    def get_mixin_key(orig: str) -> str:
        mixin_key_enc_tab = [
            46, 47, 18, 2, 53, 8, 23, 32, 15, 50, 10, 31, 58, 3, 45, 35, 27, 43, 5, 49,
            33, 9, 42, 19, 29, 28, 14, 39, 12, 38, 41, 13, 37, 48, 7, 16, 24, 55, 40,
            61, 26, 17, 0, 1, 60, 51, 30, 4, 22, 25, 54, 21, 56, 59, 6, 63, 57, 62, 11,
            36, 20, 34, 44, 52
        ]
        return ''.join([orig[i] for i in mixin_key_enc_tab])[:32]

    def get_wbi_keys() -> tuple:
        """获取 WBI keys，顺便让 Session 拿到初始的访客 Cookie"""
        resp = session.get("https://api.bilibili.com/x/web-interface/nav", proxies=proxies)
        resp.raise_for_status()
        json_content = resp.json()
        img_url = json_content['data']['wbi_img']['img_url']
        sub_url = json_content['data']['wbi_img']['sub_url']
        img_key = img_url.rsplit('/', 1)[1].split('.')[0]
        sub_key = sub_url.rsplit('/', 1)[1].split('.')[0]
        return img_key, sub_key

    def sign_params_for_wbi(params: dict, img_key: str, sub_key: str) -> dict:
        mixin_key = get_mixin_key(img_key + sub_key)
        curr_time = round(time.time())
        params['wts'] = curr_time
        sorted_params = dict(sorted(params.items()))

        encoded_parts = []
        for k, v in sorted_params.items():
            v = str(v)
            filtered_value = ''.join(filter(lambda chr: chr not in "!'()*", v))
            encoded_value = urllib.parse.quote(filtered_value, safe='')
            encoded_parts.append(f"{k}={encoded_value}")

        query = '&'.join(encoded_parts)
        wbi_sign = md5((query + mixin_key).encode()).hexdigest()
        params['w_rid'] = wbi_sign
        return params

    # --- 主爬取逻辑 ---
    print(f"准备匿名查询用户 mid={mid} 的视频，目标数量: {desired_count}...")

    try:
        img_key, sub_key = get_wbi_keys()
    except Exception as e:
        print(f"初始化 WBI Keys 失败，可能 IP 已被风控: {e}")
        return []

    collected_videos = []
    current_page = 1
    page_size = 40

    while len(collected_videos) < desired_count:
        print(f"正在获取第 {current_page} 页数据...")

        unsigned_params = {
            'mid': mid, 'order': order, 'tid': 0,
            'keyword': keyword, 'pn': current_page, 'ps': page_size,
        }
        signed_params = sign_params_for_wbi(unsigned_params, img_key, sub_key)

        try:
            url = "https://api.bilibili.com/x/space/wbi/arc/search"
            # 使用 session 发送请求，附带 proxies 参数
            response = session.get(url, params=signed_params, proxies=proxies)
            response.raise_for_status()
            result = response.json()

            if result.get("code") != 0:
                print(f"接口报错，错误码: {result.get('code')}, 信息: {result.get('message')}")
                # 如果遇到 -412 等风控码，说明当前 IP 或访客身份被限制了
                break

            data = result.get("data")
            if not data:
                break

            new_videos = data.get('list', {}).get('vlist', [])
            if not new_videos:
                print("当前页没有更多视频，已到底部。")
                break

            collected_videos.extend(new_videos)

            total_server_count = data.get('page', {}).get('count', 0)
            if len(collected_videos) >= total_server_count:
                print("已获取该用户所有公开视频。")
                break

            current_page += 1

            # 随机休眠防风控机制
            sleep_time = random.uniform(2.5, 4.5)
            print(f"获取成功，随机休眠 {sleep_time:.2f} 秒，模拟人类行为...")
            time.sleep(sleep_time)

        except Exception as e:
            print(f"查询请求发生网络或未知错误: {e}")
            break

    final_result = collected_videos[:desired_count]
    print(f"获取完成，共收集到 {len(final_result)} 个视频。")
    return final_result


import math


def calculate_video_scores(video_list, current_timestamp, windows=(6, 12, 24, 36, 48, 100000)):
    need_filed_list = ['created', 'play', 'comment', 'title', 'bvid', 'author', 'mid', 'length']
    if not video_list:
        return []

    # 计算平均每天投递视频数（总发布数 / 时间跨度天数）
    # 计算最近3天的平均每天投递视频数
    three_days_seconds = 3 * 24 * 3600
    recent_video_count = sum(1 for v in video_list if v['created'] >= current_timestamp - three_days_seconds)
    avg_daily_videos = recent_video_count / 3.0

    # 只保留必要的字段，多余字段删除
    for v in video_list:
        for key in list(v.keys()):
            if key not in need_filed_list:
                del v[key]
        v['duration'] = time_to_ms(v.get('length', '0:00')) / 1000.0  # 转换为秒，方便后续分析

    # 1. 计算基础指标：存活时间、每分钟播放速率、时间归一化后的绝对分
    for v in video_list:
        alive_minutes = max((current_timestamp - v['created']) / 60.0, 1.0)
        v['alive_hours'] = alive_minutes / 60.0
        v['play_rate'] = v['play'] / v['alive_hours']

        # 绝对得分：用每分钟播放量（已除以存活时间）取对数，彻底消除老视频的时间红利
        v['abs_score'] = math.log(v['play_rate'] + 1)

    # 2. 计算各个时间窗口的大盘平均每分钟播放量
    window_avgs = {}
    for w in windows:
        # 找出属于该窗口的视频（利用包含关系：<=24h 必然包含 <=12h）
        rates_in_window = [v['play_rate'] for v in video_list if v['alive_hours'] <= w]
        window_avgs[w] = sum(rates_in_window) / len(rates_in_window) if rates_in_window else 0.0

    # 3. 计算多周期比较得分及最终总分
    # 引入平滑系数，避免分母极小时导致 comp_score 爆炸
    # 设为 1.0 表示“每小时额外1个播放量”的保底基准，可根据实际业务水位调整
    smooth_factor = 1.0

    for v in video_list:
        ratios = []
        v['window_ratios'] = {}  # 记录中间过程，方便排查和分析数据

        # 将当前视频的速率与每个周期的平均速率分别比较
        for w in windows:
            # 要求视频必须在该窗口内才有资格比较，否则就不计算这个周期的得分了（因为不公平）
            if v['alive_hours'] > w:
                continue

            avg_rate = window_avgs[w]

            # 使用拉普拉斯平滑计算倍率
            ratio = (v['play_rate'] + smooth_factor) / (avg_rate + smooth_factor)

            ratios.append(ratio)
            v['window_ratios'][f'{w}h_ratio'] = ratio

        # 比较得分：所有周期表现比例的平均值
        v['comp_score'] = sum(ratios)

        # 最终综合得分：绝对爆发力 + 相对表现力
        v['score'] = v['abs_score'] * v['comp_score']

        # 增加平均每天发布视频数量信息
        v['avg_daily_videos'] = avg_daily_videos
        # 增加'current_time_str'
        v['current_time_str'] = datetime.fromtimestamp(current_timestamp).strftime("%Y-%m-%d %H:%M:%S")

    # 按照综合得分排序，得分高的排在前面
    video_list.sort(key=lambda x: x['score'], reverse=True)
    return video_list

def process_single_user(uid, max_hour=24):
    """
    单个用户的处理，拉取数据并且打分
    :param uid:
    :return:
    """
    all_video_file = r'W:\project\python_project\auto_video\config\all_bili_video.json'
    all_video_info = read_json(all_video_file)
    exist_video_info = all_video_info.get(str(uid), {})
    exist_video_list = exist_video_info.get('video_list', [])
    update_time = exist_video_info.get('update_time', 0)
    # calculate_video_scores(exist_video_list, current_timestamp=update_time)
    # 如果在1天内更新就不拉取新数据了
    if time.time() - update_time < max_hour * 3600:
        print(f"用户 {uid} 的视频数据在一天内已经更新过了，跳过拉取新数据。")
        return True


    videos = get_user_videos_public(mid=uid, desired_count=40)
    min_created_timestamp = 1000000000000
    if videos:
        for video_info in videos:
            created_timestamp = video_info.get('created')
            if min_created_timestamp > created_timestamp:
                min_created_timestamp = created_timestamp

        # 保留exist_video_list 中在 min_created_timestamp 之前的视频
        exist_video_list = [v for v in exist_video_list if v.get('created', 0) < min_created_timestamp]

        # 将新的视频信息添加到 exist_video_list 中
        exist_video_list.extend(videos)
        exist_video_info['video_list'] = exist_video_list
        # 增加更新时间
        exist_video_info['update_time'] = int(time.time())

        exist_video_info['update_time_str'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        all_video_info[str(uid)] = exist_video_info
        save_json(all_video_file, all_video_info)
        return True
    return False


def process_single_tag(tag, max_hour=24):
    start_time = time.time()
    all_user_file = r'W:\project\python_project\auto_video\config\all_user_info.json'
    all_user_info = read_json(all_user_file)

    exist_video_info = all_user_info.get(str(tag), {})
    video_info_list = exist_video_info.get('video_info_list', [])
    update_time = exist_video_info.get('update_time', 0)
    if time.time() - update_time < max_hour * 3600 and video_info_list:
        print(f"用户 {tag} 的视频数据在一天内已经更新过了，跳过拉取新数据。")
        return

    videos = fetch_from_search(tag, recent_days=2)
    exist_user_list = videos
    need_filed_list = ['author', 'mid', 'aid', 'bvid', 'title', 'description', 'play', 'pubdate']
    for v in videos:
        for key in list(v.keys()):
            if key not in need_filed_list:
                del v[key]

    print(f"搜索标签 {tag} 得到 {len(videos)} 个相关视频。 耗时 {time.time() - start_time:.2f} 秒。 当前时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if exist_user_list:
        exist_video_info['video_info_list'] = exist_user_list
        exist_video_info['update_time'] = int(time.time())
        all_user_info[str(tag)] = exist_video_info

        save_json(all_user_file, all_user_info)
    else:
        # 没有数据就说明被风控了，等待10s
        print(f"搜索标签 {tag} 没有得到数据，可能被风控了，等待10秒后再试...")
        time.sleep(10)



def search_good_user():
    """
    按照热门关键词进行搜索得到很多用户
    :return:
    """
    hot_tags_info = read_json(RECENT_HOT_TAGS_FILE)
    result_hot_tags = {}
    for video_type, tag_info_list in hot_tags_info.items():
        combined_tags = {}
        # 1. 遍历并累加同名标签的值
        for tag_dict in tag_info_list:
            for tag, count in tag_dict.items():
                # 如果键不存在默认给 0，然后再加 count
                combined_tags[tag] = combined_tags.get(tag, 0) + count
        sorted_tags = dict(sorted(combined_tags.items(), key=lambda item: item[1], reverse=True))
        result_hot_tags[video_type] = sorted_tags



    index_count = 0
    for video_type, sorted_tags in result_hot_tags.items():
        print(f"视频类型: {video_type}")
        for tag, count in sorted_tags.items():
            index_count += 1
            print(f"\n\n标签: {tag}, 出现次数: {count} 进度: {index_count} / {len(sorted_tags)}")
            # 可以在这里对每个标签进行搜索，获取相关视频和用户信息
            process_single_tag(tag)


def get_all_user_video_info():
    """
    获取所有用户的视频信息
    :return:
    """
    all_user_file = r'W:\project\python_project\auto_video\config\all_user_info.json'
    all_user_info = read_json(all_user_file)
    all_mid_list = []
    for tag, video_info in all_user_info.items():
        video_info_list = video_info.get('video_info_list', [])
        for video in video_info_list:
            pubdate = video.get('pubdate', 0)
            # 只保留最近两天的视频
            if time.time() - pubdate < 2 * 24 * 3600:
                all_mid_list.append(video.get('mid'))

    success_count = 0
    fail_count = 0
    all_mid_list = list(set(all_mid_list))
    print(f"从所有标签的视频信息中提取到 {len(all_mid_list)} 个唯一用户 mid。")
    for index, mid in enumerate(all_mid_list):
        try:
            print(f"\n\n正在处理用户 mid: {mid} 进度: {index + 1} / {len(all_mid_list)} 当前失败和成功数量: {fail_count} / {success_count} 当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            result = process_single_user(mid)
            if result:
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            traceback.print_exc()
            print(f"处理用户 mid: {mid} 发生异常: {e}")
            fail_count += 1
            continue

def filter_good_user():
    """
    筛选得到高频更新的用户，以及计算每个视频的得分
    :return:
    """
    all_video_file = r'W:\project\python_project\auto_video\config\all_bili_video.json'
    all_video_info = read_json(all_video_file)
    all_video_score_list = []
    process_count = 0
    for uid, exist_video_info in all_video_info.items():
        exist_video_list = exist_video_info.get('video_list', [])
        update_time = exist_video_info.get('update_time', 0)
        # 判断是否在一天内更新过了，如果没有更新过了就说明这个用户可能不活跃了，或者被风控了，就不计算分数了
        if time.time() - update_time > 24 * 3600:
            continue

        video_score_list = calculate_video_scores(exist_video_list, current_timestamp=update_time)
        process_count += 1
        all_video_score_list.extend(video_score_list)
    print(f"共处理了 {process_count} 个用户的视频数据，计算得到 {len(all_video_score_list)} 个视频的分数。")

    # 保留'avg_daily_videos' 大于1的视频，说明这个用户最近三天平均每天发布了超过1个视频，比较活跃
    all_video_score_list = [v for v in all_video_score_list if v.get('avg_daily_videos', 0) > 1.0]

    # 只保留duration 小于10分钟的视频，说明这个视频比较短，更容易爆发
    all_video_score_list = [v for v in all_video_score_list if v.get('duration', 0) < 600.0]


    # 按照分数排序，得分高的排在前面
    all_video_score_list.sort(key=lambda x: x['score'], reverse=True)
    print()



# --- 测试代码 ---
if __name__ == "__main__":
    # filter_good_user()

    get_all_user_video_info()
    # search_good_user()
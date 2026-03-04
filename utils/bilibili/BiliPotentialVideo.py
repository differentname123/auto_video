import math
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

from utils.common_utils import read_json, save_json

# 准备几个常见的 User-Agent
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) Gecko/20100101 Firefox/128.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15"
]


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


def calculate_video_scores(video_list, current_timestamp, windows=(6, 12, 24, 36, 48)):
    need_filed_list = ['created', 'play', 'comment', 'title', 'bvid']

    if not video_list:
        return []

    # 只保留必要的字段，多余字段删除
    for v in video_list:
        for key in list(v.keys()):
            if key not in need_filed_list:
                del v[key]

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
    for v in video_list:
        ratios = []
        v['window_ratios'] = {}  # 记录中间过程，方便排查和分析数据

        # 将当前视频的速率与每个周期的平均速率分别比较
        for w in windows:
            avg_rate = window_avgs[w]
            ratio = v['play_rate'] / avg_rate if avg_rate > 0 else 1.0
            ratios.append(ratio)
            v['window_ratios'][f'{w}h_ratio'] = ratio

        # 比较得分：所有周期表现比例的平均值
        v['comp_score'] = sum(ratios) / len(ratios)

        # 最终综合得分：绝对爆发力 + 相对表现力
        v['score'] = v['abs_score'] + v['comp_score']

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
    calculate_video_scores(exist_video_list, current_timestamp=update_time)
    # 如果在1天内更新就不拉取新数据了
    if time.time() - update_time < max_hour * 3600:
        print(f"用户 {uid} 的视频数据在一天内已经更新过了，跳过拉取新数据。")
        return


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


# --- 测试代码 ---
if __name__ == "__main__":
    # 以影视飓风 (mid: 946974) 为例，获取最新 5 个视频
    test_mid = 149425572
    process_single_user(test_mid)
    #
    # videos = get_user_videos_public(mid=test_mid, desired_count=40)
    # for v in videos:
    #     print(f"标题: {v.get('title')[:20]}... | BVID: {v.get('bvid')} | 播放量: {v.get('play')}")
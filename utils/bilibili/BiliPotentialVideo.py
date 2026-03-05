import traceback
import random
import math
from datetime import datetime
import requests
import urllib.parse
import time
from hashlib import md5

# 移除未使用和重复的导入
from application.video_common_config import ALL_TARGET_TAGS_INFO_FILE, RECENT_HOT_TAGS_FILE
from utils.bilibili.bili_utils import fetch_from_search
from utils.common_utils import read_json, save_json, time_to_ms

# === 优化：提取公共文件路径配置 ===
ALL_VIDEO_FILE = r'W:\project\python_project\auto_video\config\all_bili_video.json'
ALL_USER_FILE = r'W:\project\python_project\auto_video\config\all_user_info.json'

def get_user_videos_public(mid: int, desired_count: int = 30, order: str = 'pubdate', keyword: str = '',
                           use_proxy: bool = False, proxies: dict = None) -> list:
    """
    独立且免(登录)Cookie的B站用户视频获取函数 - 并发防风控版
    """

    # 【核心优化】：身份池。将配套的 UA、Headers、Cookie 和 显卡指纹强绑定
    # 警告：高并发时，请务必将 Profile 2 和 3 中的 Cookie 替换为您从无痕浏览器中抓取的真实有效 Cookie
    PROFILES = [
        {
            # 身份 1：Windows Chrome 145 (您的原生抓包数据)
            "headers": {
                "accept": "*/*", "accept-language": "zh-CN,zh;q=0.9", "origin": "https://space.bilibili.com",
                "priority": "u=1, i", "sec-fetch-dest": "empty", "sec-fetch-mode": "cors",
                "sec-fetch-site": "same-site",
                "sec-ch-ua": '"Not:A-Brand";v="99", "Google Chrome";v="145", "Chromium";v="145"',
                "sec-ch-ua-mobile": "?0", "sec-ch-ua-platform": '"Windows"',
                "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36",
                "cookie": "buvid3=A3C5C7F7-CE22-2319-1BF1-5A12667F16F681690infoc; b_nut=1772599781; _uuid=828F87AF-9108E-7BDB-BFD6-1F749476181182401infoc; home_feed_column=5; browser_resolution=1862-925; buvid4=CCC31F1C-EBCF-9C0D-9306-7D78D3D3BBDD82499-026030412-C733j3GbXXTp8uVzKptMqg%3D%3D; buvid_fp=02a64a5530e27a720f537b223dbf6381; sid=6uas0qcc; CURRENT_QUALITY=0; rpdid=|(k))kmuk|RY0J'u~~klmm)~|; bili_ticket=eyJhbGciOiJIUzI1NiIsImtpZCI6InMwMyIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3NzI4ODM3ODYsImlhdCI6MTc3MjYyNDUyNiwicGx0IjotMX0.eg7x6QGwKAc_lBgDT7tTVViek_I9b7y8fI4_eoa0OO4; bili_ticket_expires=1772883726; PVID=1; LIVE_BUVID=AUTO7417726245881683; CURRENT_FNVAL=2000; b_lsid=ED43E64E_19CBAB38E4B"
            },
            "dm_params": {
                "platform": "web", "web_location": "333.1387", "dm_img_list": "[]",
                "dm_img_str": "V2ViR0wgMS4wIChPcGVuR0wgRVMgMi4wIENocm9taXVtKQ",
                "dm_cover_img_str": "QU5HTEUgKE5WSURJQSwgTlZJRElBIEdlRm9yY2UgUlRYIDIwODAgVGkgKDB4MDAwMDFFMDcpIERpcmVjdDNEMTEgdnNfNV8wIHBzXzVfMCwgRDNEMTEpR29vZ2xlIEluYy4gKE5WSURJQS",
                "dm_img_inter": '{"ds":[],"wh":[4004,2418,100],"of":[157,314,157]}'
            }
        }
    ]

    if use_proxy and proxies is None:
        proxies = random.choice([
            {"http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"},
            {"http": "http://115.190.54.74:8888", "https": "http://115.190.54.74:8888"}
        ])

    # 随机抽取一个身份档案进行请求伪装
    profile = random.choice(PROFILES)
    session = requests.Session()
    session.headers.update(profile["headers"])
    session.headers["referer"] = f"https://space.bilibili.com/{mid}/upload/video"

    def get_mixin_key(orig: str) -> str:
        mixin_key_enc_tab = [
            46, 47, 18, 2, 53, 8, 23, 32, 15, 50, 10, 31, 58, 3, 45, 35, 27, 43, 5, 49,
            33, 9, 42, 19, 29, 28, 14, 39, 12, 38, 41, 13, 37, 48, 7, 16, 24, 55, 40,
            61, 26, 17, 0, 1, 60, 51, 30, 4, 22, 25, 54, 21, 56, 59, 6, 63, 57, 62, 11,
            36, 20, 34, 44, 52
        ]
        return ''.join([orig[i] for i in mixin_key_enc_tab])[:32]

    def get_wbi_keys() -> tuple:
        resp = session.get("https://api.bilibili.com/x/web-interface/nav", proxies=proxies, timeout=5)
        resp.raise_for_status()
        wbi_img = resp.json().get('data', {}).get('wbi_img', {})
        return wbi_img.get('img_url', '').rsplit('/', 1)[1].split('.')[0], \
        wbi_img.get('sub_url', '').rsplit('/', 1)[1].split('.')[0]

    def sign_params_for_wbi(params: dict, img_key: str, sub_key: str) -> dict:
        mixin_key = get_mixin_key(img_key + sub_key)
        params['wts'] = round(time.time())
        sorted_params = dict(sorted(params.items()))

        encoded_parts = []
        for k, v in sorted_params.items():
            filtered_value = ''.join(filter(lambda chr: chr not in "!'()*", str(v)))
            encoded_parts.append(f"{k}={urllib.parse.quote(filtered_value, safe='')}")

        query = '&'.join(encoded_parts)
        params['w_rid'] = md5((query + mixin_key).encode()).hexdigest()
        return params

    print(f"准备查询用户 mid={mid} 的视频，目标数量: {desired_count}... proxy: {use_proxy}")

    try:
        img_key, sub_key = get_wbi_keys()
    except Exception as e:
        print(f"初始化 WBI Keys 失败，可能 IP/Cookie 已被风控: {e}")
        return []

    collected_videos = []
    current_page = 1
    page_size = 40

    while len(collected_videos) < desired_count:
        print(f"正在获取第 {current_page} 页数据...")

        # 将动态参数与当前 profile 绑定的显卡指纹参数合并 (使用 ** 字典解包，使代码极其简洁)
        unsigned_params = {
            'pn': current_page, 'ps': page_size, 'tid': 0, 'special_type': '',
            'order': order, 'mid': mid, 'index': 0, 'keyword': keyword, 'order_avoided': 'true',
            **profile["dm_params"]
        }

        signed_params = sign_params_for_wbi(unsigned_params, img_key, sub_key)

        try:
            url = "https://api.bilibili.com/x/space/wbi/arc/search"
            response = session.get(url, params=signed_params, proxies=proxies, timeout=10)
            response.raise_for_status()
            result = response.json()

            if result.get("code") != 0:
                print(f"接口报错，错误码: {result.get('code')}, 信息: {result.get('message')}")
                break

            data = result.get("data")
            if not data: break

            new_videos = data.get('list', {}).get('vlist', [])
            if not new_videos:
                print("当前页没有更多视频，已到底部。")
                break

            collected_videos.extend(new_videos)

            if len(collected_videos) >= data.get('page', {}).get('count', 0):
                print("已获取该用户所有公开视频。")
                break

            current_page += 1
            # sleep_time = random.uniform(2.5, 4.5)
            # print(f"获取成功，随机休眠 {sleep_time:.2f} 秒，模拟人类行为...")
            # time.sleep(sleep_time)

        except Exception as e:
            print(f"查询请求发生网络或未知错误: {e}")
            break

    final_result = collected_videos[:desired_count]
    print(f"获取完成，共收集到 {len(final_result)} 个视频。")
    return final_result


def calculate_video_scores(video_list, current_timestamp, windows=(6, 12, 24, 36, 48, 100000)):
    need_filed_list = ['created', 'play', 'comment', 'title', 'bvid', 'author', 'mid', 'length']
    if not video_list:
        return []

    three_days_seconds = 3 * 24 * 3600
    recent_video_count = sum(1 for v in video_list if v.get('created', 0) >= current_timestamp - three_days_seconds)
    avg_daily_videos = recent_video_count / 3.0

    for v in video_list:
        # 优化：更清晰的字典过滤方式
        keys_to_del = [k for k in v if k not in need_filed_list]
        for k in keys_to_del:
            del v[k]
        v['duration'] = time_to_ms(v.get('length', '0:00')) / 1000.0

    for v in video_list:
        alive_minutes = max((current_timestamp - v['created']) / 60.0, 1.0)
        v['alive_hours'] = alive_minutes / 60.0
        v['play_rate'] = v['play'] / v['alive_hours']
        v['abs_score'] = math.log(v['play_rate'] + 1)

    window_avgs = {}
    for w in windows:
        rates_in_window = [v['play_rate'] for v in video_list if v.get('alive_hours', 0) <= w]
        window_avgs[w] = sum(rates_in_window) / len(rates_in_window) if rates_in_window else 0.0

    smooth_factor = 1.0

    for v in video_list:
        ratios = []
        v['window_ratios'] = {}

        for w in windows:
            if v['alive_hours'] > w:
                continue
            avg_rate = window_avgs[w]
            ratio = (v['play_rate'] + smooth_factor) / (avg_rate + smooth_factor)
            ratios.append(ratio)
            v['window_ratios'][f'{w}h_ratio'] = ratio

        v['comp_score'] = sum(ratios) if ratios else 0
        v['score'] = math.log(v['abs_score'] + 1) * math.log(v['comp_score'] + 1)
        v['avg_daily_videos'] = avg_daily_videos
        v['current_time_str'] = datetime.fromtimestamp(current_timestamp).strftime("%Y-%m-%d %H:%M:%S")

    video_list.sort(key=lambda x: x['score'], reverse=True)
    return video_list


def process_single_user(uid, all_video_info, max_hour=24):
    exist_video_info = all_video_info.get(str(uid), {})
    exist_video_list = exist_video_info.get('video_list', [])
    update_time = exist_video_info.get('update_time', 0)

    if time.time() - update_time < max_hour * 3600:
        print(f"用户 {uid} 的视频数据在一天内已经更新过了，跳过拉取新数据。")
        return 0

    use_proxy = random.choice([True, False])
    videos = get_user_videos_public(mid=uid, desired_count=40, use_proxy=use_proxy)
    min_created_timestamp = 1000000000000

    if videos:
        for video_info in videos:
            created_timestamp = video_info.get('created', 0)
            if min_created_timestamp > created_timestamp:
                min_created_timestamp = created_timestamp

        exist_video_list = [v for v in exist_video_list if v.get('created', 0) < min_created_timestamp]
        exist_video_list.extend(videos)

        exist_video_info['video_list'] = exist_video_list
        exist_video_info['update_time'] = int(time.time())
        exist_video_info['update_time_str'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        all_video_info[str(uid)] = exist_video_info
        # 使用常量替换硬编码
        save_json(ALL_VIDEO_FILE, all_video_info)
        return 1
    return -1


def process_single_tag(tag, all_user_info, max_hour=24):
    start_time = time.time()
    exist_video_info = all_user_info.get(str(tag), {})
    video_info_list = exist_video_info.get('video_info_list', [])
    update_time = exist_video_info.get('update_time', 0)

    if time.time() - update_time < max_hour * 3600 and video_info_list:
        print(f"用户 {tag} 的视频数据在一天内已经更新过了，跳过拉取新数据。")
        return

    videos = fetch_from_search(tag, recent_days=2)
    need_filed_list = ['author', 'mid', 'aid', 'bvid', 'title', 'description', 'play', 'pubdate']

    for v in videos:
        keys_to_del = [k for k in v if k not in need_filed_list]
        for k in keys_to_del:
            del v[k]

    print(
        f"搜索标签 {tag} 得到 {len(videos)} 个相关视频。 耗时 {time.time() - start_time:.2f} 秒。 当前时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if videos:  # 优化：去掉多余的 exist_user_list 变量
        exist_video_info['video_info_list'] = videos
        exist_video_info['update_time'] = int(time.time())
        all_user_info[str(tag)] = exist_video_info
        save_json(ALL_USER_FILE, all_user_info)
    else:
        print(f"搜索标签 {tag} 没有得到数据，可能被风控了，等待10秒后再试...")
        time.sleep(10)


def load_pure_user_info():
    all_user_info = read_json(ALL_USER_FILE)
    need_filed_list = ['author', 'mid', 'aid', 'bvid', 'title', 'description', 'play', 'pubdate']

    for tag, video_info in all_user_info.items():
        video_info_list = video_info.get('video_info_list', [])
        if not video_info_list:
            continue  # 修复：原代码是 return []，会导致整个解析因为一个空列表而异常终止

        for v in video_info_list:
            keys_to_del = [k for k in v if k not in need_filed_list]
            for k in keys_to_del:
                del v[k]

    save_json(ALL_USER_FILE, all_user_info)
    return all_user_info


def search_good_user():
    hot_tags_info = read_json(RECENT_HOT_TAGS_FILE)
    result_hot_tags = {}
    for video_type, tag_info_list in hot_tags_info.items():
        combined_tags = {}
        for tag_dict in tag_info_list:
            for tag, count in tag_dict.items():
                combined_tags[tag] = combined_tags.get(tag, 0) + count
        sorted_tags = dict(sorted(combined_tags.items(), key=lambda item: item[1], reverse=True))
        result_hot_tags[video_type] = sorted_tags

    all_user_info = load_pure_user_info()
    index_count = 0
    for video_type, sorted_tags in result_hot_tags.items():
        print(f"视频类型: {video_type}")
        for tag, count in sorted_tags.items():
            index_count += 1
            print(f"\n\n标签: {tag}, 出现次数: {count} 进度: {index_count} / {len(sorted_tags)}")
            process_single_tag(tag, all_user_info)


def load_pure_video_info():
    all_video_info = read_json(ALL_VIDEO_FILE)
    need_filed_list = ['created', 'play', 'title', 'bvid', 'author', 'mid', 'length']

    for uid, exist_video_info in all_video_info.items():
        exist_video_list = exist_video_info.get('video_list', [])
        if not exist_video_list:
            continue  # 修复：同上，改为 continue

        for v in exist_video_list:
            keys_to_del = [k for k in v if k not in need_filed_list]
            for k in keys_to_del:
                del v[k]

    save_json(ALL_VIDEO_FILE, all_video_info)
    return all_video_info


def get_all_user_video_info():
    all_user_info = read_json(ALL_USER_FILE)
    all_mid_list = []

    for tag, video_info in all_user_info.items():
        video_info_list = video_info.get('video_info_list', [])
        for video in video_info_list:
            pubdate = video.get('pubdate', 0)
            if time.time() - pubdate < 2 * 24 * 3600:
                all_mid_list.append(video.get('mid'))

    success_count = 0
    fail_count = 0
    all_mid_list = list(set(all_mid_list))
    print(f"从所有标签的视频信息中提取到 {len(all_mid_list)} 个唯一用户 mid。")
    all_video_info = load_pure_video_info()

    for index, mid in enumerate(all_mid_list):
        try:
            print(
                f"\n\n正在处理用户 mid: {mid} 进度: {index + 1} / {len(all_mid_list)} 当前失败和成功数量: {fail_count} / {success_count} 当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            result = process_single_user(mid, all_video_info)
            if result == 1:
                success_count += 1
            elif result == -1:
                fail_count += 1
        except Exception as e:
            traceback.print_exc()
            print(f"处理用户 mid: {mid} 发生异常: {e}")
            fail_count += 1
            continue


def filter_good_user():
    all_video_info = read_json(ALL_VIDEO_FILE)
    all_video_score_list = []
    process_count = 0

    for uid, exist_video_info in all_video_info.items():
        exist_video_list = exist_video_info.get('video_list', [])
        update_time = exist_video_info.get('update_time', 0)

        if time.time() - update_time > 24 * 3600:
            continue

        video_score_list = calculate_video_scores(exist_video_list, current_timestamp=update_time)
        process_count += 1
        all_video_score_list.extend(video_score_list)

    print(f"共处理了 {process_count} 个用户的视频数据，计算得到 {len(all_video_score_list)} 个视频的分数。")

    all_video_score_list = [v for v in all_video_score_list if v.get('avg_daily_videos', 0) > 1.0]
    all_video_score_list = [v for v in all_video_score_list if v.get('duration', 0) < 600.0]

    all_video_score_list.sort(key=lambda x: x['score'], reverse=True)

    # 优化：原代码末尾只是 print()，补充返回排序结果
    print(f"筛选完成，当前共有 {len(all_video_score_list)} 个符合条件的高分视频。")
    return all_video_score_list


# --- 测试代码 ---
if __name__ == "__main__":
    # filter_good_user()
    while True:
        try:
            get_all_user_video_info()
            # search_good_user()
        except Exception as e:
            traceback.print_exc()
            print(f"主循环发生异常: {e}")
        print("等待30秒后重试...")
        time.sleep(30)
import traceback
import random
import math
from datetime import datetime
import requests
import urllib.parse
import time
from hashlib import md5
import concurrent.futures  # 新增：用于多线程并发
import threading  # 新增：用于共享资源的线程锁保护

# 移除未使用和重复的导入
from application.video_common_config import ALL_TARGET_TAGS_INFO_FILE, RECENT_HOT_TAGS_FILE
from utils.bilibili.bili_utils import fetch_from_search
from utils.bilibili.get_bilibili_real_profile import get_bilibili_fresh_profile
from utils.common_utils import read_json, save_json, time_to_ms, read_file_to_str, string_to_object
from utils.gemini import get_llm_content_gemini_flash_video, get_llm_content

# === 优化：提取公共文件路径配置 ===
ALL_VIDEO_FILE = r'W:\project\python_project\auto_video\config\all_bili_video.json'
ALL_USER_FILE = r'W:\project\python_project\auto_video\config\all_user_info.json'

ALL_GOOD_USER_FILE = r'W:\project\python_project\auto_video\config\all_good_user_info.json'

ALL_USER_TYPE_MAP_FILE = r'W:\project\python_project\auto_video\config\all_user_type_map.json'
ALL_GOOD_VIDEO_FILE = r'W:\project\python_project\auto_video\config\all_good_video.json'
BASE_PROFILES = [
    {
        # 身份 3：基于您捕捉的 Firefox 148 原生抓包数据进行严格匹配
        "headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:148.0) Gecko/20100101 Firefox/148.0",
            "Accept": "*/*",
            "Accept-Language": "zh-CN,zh;q=0.9,zh-TW;q=0.8,zh-HK;q=0.7,en-US;q=0.6,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Origin": "https://space.bilibili.com",
            "Sec-GPC": "1",
            "Connection": "keep-alive",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-site",
            "Priority": "u=4",
            "TE": "trailers",  # 补充了抓包中存在的 TE 头
            # 【修改】使用 curl 中完全一致的 Cookie
            "Cookie": "buvid3=ADC48BBC-C48D-32BD-6E56-4BFB55D0DE9603414infoc; b_nut=1772839403; __at_once=18129603945248101006; buvid4=B47E505C-ED4B-8166-D3D2-A6A677005C3404828-026030707-tPguV4f7Z30ul7OW%2FTU71Q%3D%3D; bili_ticket=eyJhbGciOiJIUzI1NiIsImtpZCI6InMwMyIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3NzMwOTg2MDUsImlhdCI6MTc3MjgzOTM0NSwicGx0IjotMX0.ZzJwVNy04i8RjUhw9kpXI_CBY-n2vfmL8BWtEfKIgGU; bili_ticket_expires=1773098545; buvid_fp=446700a5f2f4f41f3e38bd98cabbc908; CURRENT_FNVAL=2000; sid=gqvw5gl7"
        },
        "dm_params": {
            "platform": "web",
            "web_location": "333.1387",
            # 【修改】还原抓包中真实的空轨迹数据
            "dm_img_list": "[]",
            "dm_img_str": "V2ViR0wgMS",
            "dm_cover_img_str": "QU5HTEUgKE5WSURJQSwgTlZJRElBIEdlRm9yY2UgR1RYIDk4MCBEaXJlY3QzRDExIHZzXzVfMCBwc181XzApLCBvciBzaW1pbGFyR29vZ2xlIEluYy4gKE5WSURJQS",
            # 【修改】还原抓包中真实的页面交互和环境特征数据 (已做URL解码处理)
            "dm_img_inter": '{"ds":[],"wh":[5235,7405,15],"of":[491,982,491]}'
        }
    },

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
    },
    {
        # 身份 2：基于您捕捉的 Firefox 148 原生抓包数据进行严格匹配
        "headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:148.0) Gecko/20100101 Firefox/148.0",
            "Accept": "*/*",
            "Accept-Language": "zh-CN,zh;q=0.9,zh-TW;q=0.8,zh-HK;q=0.7,en-US;q=0.6,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Origin": "https://space.bilibili.com",
            "Sec-GPC": "1",
            "Connection": "keep-alive",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-site",
            "Priority": "u=4",
            "Cookie": "buvid3=14E13BE1-9C29-B253-1E41-2B46DB27089423391infoc; b_nut=1772753823; __at_once=10441780667350004868; bili_ticket=eyJhbGciOiJIUzI1NiIsImtpZCI6InMwMyIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3NzMwMTMwMjQsImlhdCI6MTc3Mjc1Mzc2NCwicGx0IjotMX0.-xA30SvAlxLAINx_CfNFOpMWxQSBGYgic5H0IONb_ls; bili_ticket_expires=1773012964; buvid4=543C034A-6F64-159F-88B3-68DB3591885924296-026030607-tPguV4f7Z30ul7OW%2FTU71Q%3D%3D; buvid_fp=266a273e7f2ada11ebf2f497ea6a4b7b; CURRENT_FNVAL=2000; sid=5z63h9e6"
        },
        "dm_params": {
            "platform": "web",
            "web_location": "333.1387",
            "dm_img_list": "[]",
            "dm_img_str": "V2ViR0wgMS",
            "dm_cover_img_str": "QU5HTEUgKE5WSURJQSwgTlZJRElBIEdlRm9yY2UgR1RYIDk4MCBEaXJlY3QzRDExIHZzXzVfMCBwc181XzApLCBvciBzaW1pbGFyR29vZ2xlIEluYy4gKE5WSURJQS",
            "dm_img_inter": '{"ds":[],"wh":[5382,7454,64],"of":[178,356,178]}'
        }
    },
    {
        # 身份 3：基于您捕捉的 Firefox 148 原生抓包数据进行严格匹配
        "headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:148.0) Gecko/20100101 Firefox/148.0",
            "Accept": "*/*",
            "Accept-Language": "zh-CN,zh;q=0.9,zh-TW;q=0.8,zh-HK;q=0.7,en-US;q=0.6,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Origin": "https://space.bilibili.com",
            "Sec-GPC": "1",
            "Connection": "keep-alive",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-site",
            "Priority": "u=4",
            "TE": "trailers",  # 补充了抓包中存在的 TE 头
            # 使用 curl 中完全一致的 Cookie
            "Cookie": "buvid3=26370685-F7D5-7985-9016-10FADE22B0D575328infoc; b_nut=1772772275; __at_once=907977295708651421; buvid4=68ADC892-0AE5-E3FC-9B0B-41599A2B83BE76184-026030612-tPguV4f7Z30ul7OW%2FTU71Q%3D%3D; bili_ticket=eyJhbGciOiJIUzI1NiIsImtpZCI6InMwMyIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3NzMwMzE0NzYsImlhdCI6MTc3Mjc3MjIxNiwicGx0IjotMX0.bNzKK01FWT7cIwli-O377Dhh-JytjJ45yiUkkygsQ9s; bili_ticket_expires=1773031416; buvid_fp=5ce8c95d90333611d4b197f1a9f09267"
        },
        "dm_params": {
            "platform": "web",
            "web_location": "333.1387",
            # 还原真实的鼠标/按键轨迹数据
            "dm_img_list": '[{"x":2026,"y":-1822,"z":0,"timestamp":7,"k":94,"type":0}]',
            "dm_img_str": "V2ViR0wgMS",
            "dm_cover_img_str": "QU5HTEUgKE5WSURJQSwgTlZJRElBIEdlRm9yY2UgR1RYIDk4MCBEaXJlY3QzRDExIHZzXzVfMCBwc181XzApLCBvciBzaW1pbGFyR29vZ2xlIEluYy4gKE5WSURJQS",
            # 还原真实的页面交互和环境特征数据
            "dm_img_inter": '{"ds":[{"t":0,"c":"bnByb2dyZXNzLWJ1c3","p":[267,89,89],"s":[52,7236,5696]}],"wh":[5487,7489,99],"of":[146,292,146]}'
        }
    },

    {
        # 身份 4：ad3
        "headers": {
            "accept": "*/*",
            "accept-language": "zh-CN,zh;q=0.9",
            "origin": "https://space.bilibili.com",
            "priority": "u=1, i",
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-site",
            "sec-ch-ua": '"Chromium";v="140", "Not=A?Brand";v="24", "Google Chrome";v="140"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36",
            "cookie": "buvid3=5A3E5C48-1334-2437-62DE-AF105F0D75C508480infoc; b_nut=1772771108; __at_once=2518939619716495421; bili_ticket=eyJhbGciOiJIUzI1NiIsImtpZCI6InMwMyIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3NzMwMzAzMDgsImlhdCI6MTc3Mjc3MTA0OCwicGx0IjotMX0.socfjRWgdc6eqW1hjV937DKmiCG0DBTOP7224bgMjuo; bili_ticket_expires=1773030248; buvid4=B7AE2A90-FDDD-90F1-BAC5-F0333949CE8709028-026030612-tPguV4f7Z30ul7OW%2FTU71Q%3D%3D; buvid_fp=bce33e466000947e3212fd8857254bbf"
        },
        "dm_params": {
            "platform": "web",
            "web_location": "333.1387",
            # 这里必须是解码后的 JSON 字符串，因为 requests 会自动进行 urlencode 编码
            "dm_img_list": '[{"x":873,"y":313,"z":0,"timestamp":31,"k":99,"type":0},{"x":1487,"y":458,"z":89,"timestamp":256,"k":115,"type":0}]',
            "dm_img_str": "V2ViR0wgMS4wIChPcGVuR0wgRVMgMi4wIENocm9taXVtKQ",
            "dm_cover_img_str": "QU5HTEUgKEludGVsLCBJbnRlbChSKSBVSEQgR3JhcGhpY3MgKDB4MDAwMDlCQTQpIERpcmVjdDNEMTEgdnNfNV8wIHBzXzVfMCwgRDNEMTEtMjcuMjAuMTAwLjg3MjkpR29vZ2xlIEluYy4gKEludGVsKQ",
            "dm_img_inter": '{"ds":[{"t":0,"c":"bnByb2dyZXNzLWJ1c3","p":[45,15,15],"s":[268,6344,2084]}],"wh":[4772,4519,22],"of":[223,446,223]}'
        }
    }
]


def get_user_videos_public(mid: int, desired_count: int = 30, order: str = 'pubdate', keyword: str = '',
                           use_proxy: bool = False, proxies: dict = None, new_profile_list=None) -> tuple: # 修改点：改变了返回提示符为 tuple
    """
    独立且免(登录)Cookie的B站用户视频获取函数 - 并发防风控版
    """

    # 【核心优化】：身份池。将配套的 UA、Headers、Cookie 和 显卡指纹强绑定
    # 警告：高并发时，请务必将 Profile 2 和 3 中的 Cookie 替换为您从无痕浏览器中抓取的真实有效 Cookie
    if new_profile_list:
        PROFILES = new_profile_list

    if use_proxy and proxies is None:
        proxies = random.choice([
            {"http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"},
            {"http": "http://115.190.54.74:8888", "https": "http://115.190.54.74:8888"},
            None
        ])

    # 随机抽取一个身份档案进行请求伪装
    # 获取随机的index 通过index获取profile
    random_index = random.randint(0, len(PROFILES) - 1)
    profile = PROFILES[random_index]
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

    # print(f"准备查询用户 mid={mid} 的视频，目标数量: {desired_count}... proxy: {use_proxy}")

    try:
        img_key, sub_key = get_wbi_keys()
    except Exception as e:
        print(f"初始化 WBI Keys 失败，可能 IP/Cookie 已被风控: {e}")
        return [], random_index  # 修改点：异常直接返回元组，包含空列表和生成的 index

    collected_videos = []
    current_page = 1
    page_size = 40

    while len(collected_videos) < desired_count:
        # print(f"正在获取第 {current_page} 页数据...")

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
                print(
                    f"mid :{mid} 接口报错，错误码: {result.get('code')}, 信息: {result.get('message')} proxies {proxies} random_index：{random_index}")
                break

            data = result.get("data")
            if not data: break

            new_videos = data.get('list', {}).get('vlist', [])
            if not new_videos:
                print(f"mid :{mid} 当前页没有更多视频，已到底部。")
                break

            collected_videos.extend(new_videos)

            if len(collected_videos) >= data.get('page', {}).get('count', 0):
                print(f"mid :{mid} 已获取该用户所有公开视频。")
                break

            current_page += 1
            # sleep_time = random.uniform(2.5, 4.5)
            # print(f"获取成功，随机休眠 {sleep_time:.2f} 秒，模拟人类行为...")
            # time.sleep(sleep_time)

        except Exception as e:
            print(f"查询请求发生网络或未知错误: proxies {proxies} random_index：{random_index}  {e}")
            break

    final_result = collected_videos[:desired_count]
    # print(f"获取完成，共收集到 {len(final_result)} 个视频。")
    return final_result, random_index  # 修改点：返回视频结果与所用的 index


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
        try:
            alive_minutes = max((current_timestamp - v['created']) / 60.0, 1.0)
            v['alive_hours'] = alive_minutes / 60.0
            v['play_rate'] = v['play'] / v['alive_hours']
            v['abs_score'] = math.log(v['play_rate'] + 1)
        except Exception as e:
            traceback.print_exc()
            print(f"计算视频 {v.get('bvid', '未知')} 的分数时发生错误: {e}")
            v['alive_hours'] = 0
            v['play_rate'] = 0
            v['abs_score'] = 0

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
        v['score'] = math.log(v['abs_score'] + 1) *  math.log(v['abs_score'] + 1) * math.log(v['comp_score'] + 1)
        v['avg_daily_videos'] = avg_daily_videos
        v['update_time_str'] = datetime.fromtimestamp(current_timestamp).strftime("%Y-%m-%d %H:%M:%S")
        v['update_time'] = current_timestamp

    video_list.sort(key=lambda x: x['score'], reverse=True)
    return video_list


def process_single_user(uid, all_video_info, data_lock, max_hour=24, new_profile_list=None):
    """
    修改点：引入 data_lock 保护读取；移除内部 save_json，交由外层调度器批量轻量保存。
    同时接收并抛出 profile_index 状态以便于外层做失败统计。
    """
    with data_lock:
        # 使用 .copy() 防御性复制，避免与主线程中 json.dumps 循环遍历时发生修改冲突
        exist_video_info = all_video_info.get(str(uid), {}).copy()
        exist_video_list = exist_video_info.get('video_list', [])
        update_time = exist_video_info.get('update_time', 0)

    if time.time() - update_time < max_hour * 3600:
        # print(f"用户 {uid} 的视频数据在一天内已经更新过了，跳过拉取新数据。")
        return 0, -1  # 修改点：返回 0 (跳过)，且 index 返回 -1 表示未分配

    # 修改点：解包接收 videos 以及所使用的 index
    videos, used_index = get_user_videos_public(mid=uid, desired_count=40, use_proxy=True, new_profile_list=new_profile_list)
    min_created_timestamp = 1000000000000

    if videos:
        for video_info in videos:
            created_timestamp = video_info.get('created', 0)
            if min_created_timestamp > created_timestamp:
                min_created_timestamp = created_timestamp

        exist_video_list = [v for v in exist_video_list if v.get('created', 0) < min_created_timestamp]
        exist_video_list.extend(videos)

        # 新增修改点：按时间戳降序排序（最新的在前），并只保留前100条
        exist_video_list.sort(key=lambda x: x.get('created', 0), reverse=True)
        exist_video_list = exist_video_list[:100]

        exist_video_info['video_list'] = exist_video_list
        exist_video_info['update_time'] = int(time.time())
        exist_video_info['update_time_str'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with data_lock:
            all_video_info[str(uid)] = exist_video_info
        # 移除 save_json，极大地提升单次处理速度
        return 1, used_index  # 修改点：一并返回 index
    return -1, used_index     # 修改点：失败时也返回 index


def process_mid_list_concurrently(all_mid_list, all_video_info, max_workers=5, save_interval=20, max_hour=24):
    """
    新增功能函数：多线程提取逻辑 & 批量落盘机制
    调整点：增加了超过200失败时最高进行5次重试的防跳过机制，并在每轮汇总打印 Profiles index 统计。
    """
    total_mids = len(all_mid_list)
    # 建立全局锁，用于保护共享大字典及写文件的安全性
    data_lock = threading.Lock()
    max_retries = 5
    fail_count = 200
    for attempt in range(1, max_retries + 1):
        start_time = time.time()
        success_count = 0
        fail_count = 0
        jump_count = 0
        processed_since_save = 0
        new_profile_list = BASE_PROFILES.copy()

        for i in range(1):
            new_profile = get_bilibili_fresh_profile()
            if new_profile:
                new_profile_list.append(new_profile)

        # 修改点：新增字典用于按轮次统计具体 index 的使用与失败情况
        profile_stats = {}

        print(
            f"\n--- 开始多线程处理 (第 {attempt}/{max_retries} 轮)，共提取 {total_mids} 个独立用户，设定最大并发线程: {max_workers} ---")

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 批量向线程池提交任务
            future_to_mid = {
                executor.submit(process_single_user, mid, all_video_info, data_lock, max_hour, new_profile_list): mid
                for mid in all_mid_list
            }

            # as_completed 只要有完成的就会 yield 输出，完美解决多线程下计数器竞态的问题
            for index, future in enumerate(concurrent.futures.as_completed(future_to_mid)):
                mid = future_to_mid[future]
                try:
                    # 修改点：解包接收两个返回值
                    result, used_index = future.result()

                    # 修改点：归档统计使用情况
                    if used_index != -1:
                        if used_index not in profile_stats:
                            profile_stats[used_index] = {'used': 0, 'failed': 0}
                        profile_stats[used_index]['used'] += 1

                    if result == 1:
                        success_count += 1
                        processed_since_save += 1
                    elif result == -1:
                        fail_count += 1
                        # 修改点：失败时增加对应 index 的失败计数
                        if used_index != -1:
                            profile_stats[used_index]['failed'] += 1
                    else:
                        jump_count += 1
                    if index % 100 == 0 or index == total_mids - 1:
                        print(f"\n正在处理用户 mid: {mid} 进度: {index + 1} / {total_mids} 当前失败和成功数量: {fail_count} / {success_count} jump_count: {jump_count} 当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

                    # 【轻量及时保存机制】：每成功保存 `save_interval` 个后，落盘一次，解决每次几百兆写入耗时的问题
                    if processed_since_save >= save_interval:
                        with data_lock:
                            save_json(ALL_VIDEO_FILE, all_video_info)
                        print(f"\n>>>> 触发批量定时保存：最新 {processed_since_save} 个用户数据已落盘 <<<<\n")
                        processed_since_save = 0

                except Exception as e:
                    traceback.print_exc()
                    print(f"处理用户 mid: {mid} 发生异常: {e}")
                    fail_count += 1

        # 处理结束扫尾工作：如果还有积攒未保存的新数据，进行最终兜底落盘
        if processed_since_save > 0:
            with data_lock:
                save_json(ALL_VIDEO_FILE, all_video_info)
            print(f"\n>>>> 触发最终扫尾保存：剩余的 {processed_since_save} 个新用户数据已落盘 <<<<\n")

        print(f"\n--- 第 {attempt} 轮多线程处理完成！总耗时: {time.time() - start_time:.2f} 秒。成功: {success_count}，失败: {fail_count}，跳过: {jump_count} ---\n")

        # 修改点：每轮执行完后，打印 Profile 详细统计
        print(">>> 📊 本轮 PROFILES (Index) 详细统计 <<<")
        if profile_stats:
            for idx in sorted(profile_stats.keys()):
                stats = profile_stats[idx]
                fail_rate = (stats['failed'] / stats['used']) * 100 if stats['used'] > 0 else 0
                print(
                    f"  Profile Index [{idx}]: 尝试次数: {stats['used']:<4} | 失败次数: {stats['failed']:<4} | 失败率: {fail_rate:.2f}%")
        else:
            print("  本轮没有使用任何 Profile (可能是用户数据都在未过期跳过范围内)。")
        print(">>> -------------------------------------- <<<\n")

        # 检查重试机制
        if fail_count <= 200:
            print(f"失败数量({fail_count})在可接受范围内(<=200)，无需重试，正常结束本阶段处理。")
            break
        else:
            if attempt < max_retries:
                print(f"注意：检测到 fail_count({fail_count}) > 200，准备触发原路重试机制，进行第 {attempt + 1} 轮执行...")
                time.sleep(10)  # 稍微睡眠避嫌，防止高频触发风控
            else:
                print(f"警告：已达到最大重试次数({max_retries}次)，fail_count 仍大于 200 ({fail_count})，强制结束本阶段。")
    return fail_count < 200

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
    all_tag_count = sum(len(tags) for tags in result_hot_tags.values())
    index_count = 0
    for video_type, sorted_tags in result_hot_tags.items():
        print(f"视频类型: {video_type}")
        for tag, count in sorted_tags.items():
            index_count += 1
            print(f"\n\n标签: {tag}, 出现次数: {count} 进度: {index_count} / {all_tag_count}")
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
    """
    修改点：旧代码直接循环遍历进行串行处理，现在交接给新提取的 process_mid_list_concurrently 进行多并发。
    """
    all_user_info = read_json(ALL_USER_FILE)
    all_mid_list = []

    for tag, video_info in all_user_info.items():
        video_info_list = video_info.get('video_info_list', [])
        for video in video_info_list:
            pubdate = video.get('pubdate', 0)
            if time.time() - pubdate < 2 * 24 * 3600:
                all_mid_list.append(video.get('mid'))

    all_mid_list = list(set(all_mid_list))
    print(f"从所有标签的视频信息中提取到 {len(all_mid_list)} 个唯一用户 mid。")
    all_video_info = load_pure_video_info()

    # 调用提取出来的多线程调度器
    # max_workers 可以根据你的代理池稳定度适当调增（默认设 5 防高频风控），save_interval 就是批量保存的阈值
    process_mid_list_concurrently(all_mid_list, all_video_info, max_workers=5, save_interval=1000)


def gen_uid_type_llm(uid_info_list):
    """
    批量获取uid的类型
    """
    log_pre = f"批量获取uid的类型 当前时间 {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}"

    retry_delay = 10
    max_retries = 5
    prompt_file_path = r'W:\project\python_project\auto_video\application\prompt\视频作者类型判断.txt'
    prompt = read_file_to_str(prompt_file_path)
    full_prompt = f'{prompt}'
    full_prompt += f'\n{uid_info_list}'
    raw = ""
    for attempt in range(1, max_retries + 1):
        try:
            model_name_list = ["gemini-2.5-flash", "gemini-3-flash-preview"]
            model_name = random.choice(model_name_list)
            print(f"批量获取uid的类型 (尝试 {attempt}/{max_retries}) {log_pre}")
            raw = get_llm_content(prompt=full_prompt, model_name=model_name)

            type_result_dict = string_to_object(raw)
            check_result, check_info = True, ""
            if not check_result:
                error_info = f"优化方案检查未通过: {check_info} {raw} {log_pre} {check_info}"
                raise ValueError(error_info)
            return type_result_dict
        except Exception as e:
            error_str = f"{str(e)} {log_pre}"
            print(f"批量获取uid的类型 (尝试 {attempt}/{max_retries}): {e} {raw} {log_pre}")
            if attempt < max_retries:
                print(f"正在重试... (等待 {retry_delay} 秒) {log_pre}")
                time.sleep(retry_delay)  # 等待一段时间后再重试
            else:
                print(f"达到最大重试次数，失败. {log_pre}")
                return None  # 达到最大重试次数后返回 None


def update_uid_type():
    """
    进行用户类型的更新，主要是依据最近发的视频然后分为 sport game 和 fun
    修改点：引入 update_time 记录，对未收录或上次更新超过 24 小时的用户进行重新判断
    """
    start_time = time.time()
    all_video_score_list, all_video_info = get_sorted_high_score_videos()

    all_user_type_info = read_json(ALL_USER_TYPE_MAP_FILE)
    # 确保字典存在
    if not isinstance(all_user_type_info, dict):
        all_user_type_info = {}

    all_video_info = load_pure_video_info()
    unique_uids = set(v['mid'] for v in all_video_score_list)

    current_time = time.time()
    uids_to_update = []

    # 1. 核心修改：找到不在 all_user_type_info 中的 uid，或者更新时间超过 24 小时的 uid
    for uid in unique_uids:
        uid_str = str(uid)
        if uid_str not in all_user_type_info:
            uids_to_update.append(uid)
        else:
            user_data = all_user_type_info[uid_str]
            # 检查新数据结构是否为字典，并判断时间
            if isinstance(user_data, dict):
                last_update = user_data.get('update_time', 0)
                if current_time - last_update > 24 * 3600:
                    uids_to_update.append(uid)
            else:
                # 兼容旧版本直接存字符串的数据，强制拉取并更新为包含时间戳的新字典结构
                uids_to_update.append(uid)

    if not uids_to_update:
        print("没有检测到需要更新类型的新用户或过期用户。")
        return

    print(f"检测到 {len(uids_to_update)} 个用户需要更新类型 (新增或已超过24小时)...")

    # 2. 对每个 uids_to_update 的uid保留最新 5 个视频的title，新建 dict
    uid_latest_titles_dict = {}
    for uid in uids_to_update:
        uid_str = str(uid)
        if uid_str in all_video_info:
            video_list = all_video_info[uid_str].get('video_list', [])
            # 按创建时间 (created) 降序排序，保证最前面的是最新视频
            sorted_videos = sorted(video_list, key=lambda x: x.get('created', 0), reverse=True)
            # 截取前 5 个视频的 title
            latest_5_titles = [v.get('title', '') for v in sorted_videos[:5]]
            uid_latest_titles_dict[uid_str] = latest_5_titles

    # 3. 对新dict进行batch_size分组，默认为50，每一组都加入batch_list
    batch_size = 50
    batch_list = []
    current_batch = {}

    for uid, titles in uid_latest_titles_dict.items():
        current_batch[uid] = titles
        if len(current_batch) == batch_size:
            batch_list.append(current_batch)
            current_batch = {}  # 重置当前批次

    # 将最后不满 50 的剩余数据加入 batch_list
    if current_batch:
        batch_list.append(current_batch)

    # 4 & 5. 遍历batch_list，调用 gen_uid_type_llm 并保存
    for index, batch in enumerate(batch_list):
        batch_start_time = time.time()
        try:
            print(f"\n\n正在处理第 {index + 1} 批次，共 {len(batch_list)} 批次，当前批次包含 {len(batch)} 个用户... 当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            type_result_dict = gen_uid_type_llm(batch)

            if type_result_dict:
                # 5. 每一个batch执行后都需要更新all_user_type_info并且保存
                for uid, v_type in type_result_dict.items():
                    if v_type in ["体育", "游戏", "娱乐", "军事"]:
                        # 核心修改：保存为字典对象，记录类型和当前的时间戳
                        all_user_type_info[str(uid)] = {
                            "type": v_type,
                            "update_time": int(time.time())
                        }

                # 覆盖保存 JSON
                save_json(ALL_USER_TYPE_MAP_FILE, all_user_type_info)
                print(f"Batch {index + 1}/{len(batch_list)} 更新完毕并保存，本批次处理 {len(type_result_dict)} 个用户。 耗时 {time.time() - batch_start_time:.2f} 秒。")
            else:
                print(f"Batch {index + 1}/{len(batch_list)} gen_uid_type_llm 返回为空，跳过保存。")

        except Exception as e:
            # 加入异常捕获防阻断，一个批次失败不影响下一个批次
            print(f"Batch {index + 1}/{len(batch_list)} 调用 LLM 或保存时发生异常: {e}")
            traceback.print_exc()

    print(f"所有批次处理完成！总耗时: {time.time() - start_time:.2f} 秒。总共更新了 {len(uids_to_update)} 个用户的类型。 批次数量: {len(batch_list)}。")


def get_sorted_high_score_videos(max_hour=24):
    """
    处理视频信息：计算得分、根据条件过滤并按得分降序排序。
    修改点：兼容读取新结构的字典型 video_type 数据，并将题材过滤提前到外层循环以提升效率并统计拦截量
    """
    all_video_info = load_pure_video_info()
    all_user_type_info = read_json(ALL_USER_TYPE_MAP_FILE)  # 修改：提前读取用户类型数据

    # 初始化统计数据
    stats = {
        "total_users": len(all_video_info),
        "filter_invalid_type": 0,  # 题材不符合要求的用户数（新增）
        "filter_update_timeout": 0,  # 更新时间超过 max_hour 的用户数
        "filter_no_recent_video": 0,  # 24小时内无新视频的用户数
        "processed_users": 0  # 最终处理的用户数
    }

    all_video_score_list = []
    current_now = time.time()  # 统一使用当前时间，避免循环中微小差异
    total_count = 0
    for uid, exist_video_info in all_video_info.items():
        exist_video_list = exist_video_info.get('video_list', [])

        total_count += len(exist_video_list)
        # 核心修改：提前获取和判断题材类型
        user_type_data = all_user_type_info.get(str(uid))
        if isinstance(user_type_data, dict):
            video_type = user_type_data.get('type', "未知")
        elif isinstance(user_type_data, str):
            video_type = user_type_data
        else:
            video_type = "未知"

        # 如果类型不在指定范围内，直接过滤并跳过后续耗时计算
        if video_type not in ['娱乐', '游戏', '体育']:
            stats["filter_invalid_type"] += 1
            continue

        update_time = exist_video_info.get('update_time', 0)

        # 过滤原因 1：更新时间太久
        if current_now - update_time > max_hour * 3600:
            stats["filter_update_timeout"] += 1
            continue

        # 要求 24 小时内有视频
        latest_created_time = max((v.get('created', 0) for v in exist_video_list), default=0)

        # 过滤原因 2：24小时内没有发布视频
        if current_now - latest_created_time > 24 * 3600:
            stats["filter_no_recent_video"] += 1
            continue

        # 成功通过过滤
        video_score_list = calculate_video_scores(exist_video_list, current_timestamp=update_time)

        # 将题材信息提前注入到经过计算的视频属性中
        for v in video_score_list:
            v['video_type'] = video_type

        stats["processed_users"] += 1
        all_video_score_list.extend(video_score_list)

    # 按得分降序排序 (根据函数名要求补充排序逻辑)
    all_video_score_list.sort(key=lambda x: x.get('score', 0), reverse=True)

    # 打印详细统计结果
    print("-" * 30)
    print(f"【处理统计报告】")
    print(f"1. 总用户数: {stats['total_users']} 过滤前总视频数: {total_count}")
    print(f"2. 过滤-题材不符 (非娱乐/游戏/体育): {stats['filter_invalid_type']}")
    print(f"3. 过滤-更新超时 (>{max_hour}h): {stats['filter_update_timeout']}")
    print(f"4. 过滤-24h内无新视频: {stats['filter_no_recent_video']}")
    print(f"5. 最终处理用户数: {stats['processed_users']}")
    print(f"6. 最终生成视频分数记录数: {len(all_video_score_list)}")
    print("-" * 30)

    # 基础过滤条件
    all_video_score_list = [v for v in all_video_score_list if v.get('avg_daily_videos', 0) > 1.0]
    all_video_score_list = [v for v in all_video_score_list if v.get('duration', 0) < 600.0]
    all_video_score_list = [v for v in all_video_score_list if v.get('alive_hours', 0) < 72]

    # 按分数降序排序
    all_video_score_list.sort(key=lambda x: x['score'], reverse=True)

    good_video_count = 1000
    type_count_map = {}
    pure_all_video_score_list = []

    # 截断或只提取需使用的字段等
    for v in all_video_score_list:
        video_type = v.get('video_type', '未知')
        if video_type not in type_count_map:
            type_count_map[video_type] = 0
        type_count_map[video_type] += 1
        if type_count_map[video_type] > good_video_count:
            continue
        pure_video_info = v.copy()
        pure_all_video_score_list.append(pure_video_info)

    save_json(ALL_GOOD_VIDEO_FILE, pure_all_video_score_list)

    return all_video_score_list, all_video_info



def update_good_user_video():
    """
    主流程：加载数据 -> 获取高分视频 -> 提取用户 -> 更新用户视频
    """

    # 1. 调用独立出来的函数获取处理后的排序列表
    all_video_score_list, all_video_info = get_sorted_high_score_videos()

    # 2. 获取不重复的uid (注：如果是自带的json库，直接存set可能会报错，建议转为list，如 list(unique_uids))
    unique_uids = set(v['mid'] for v in all_video_score_list)
    save_json(ALL_GOOD_USER_FILE, list(unique_uids))
    print(f"筛选完成，当前共有 {len(all_video_score_list)} 个符合条件的高分视频。 来源于 {len(unique_uids)} 个不同的用户。")

    max_hour = 1
    if not (5 <= datetime.now().hour < 24):
        max_hour = 2

    is_finish = process_mid_list_concurrently(unique_uids, all_video_info, max_workers=5, save_interval=1000, max_hour=max_hour)

    return is_finish


def get_good_video(video_type=None):
    """
    获取好的视频，包含video_type等信息，按照时间和分数分类，用于前端展示
    :return: dict 包含 trending (最近蹿升) 和 high_score (高分视频)
    """
    type_cn_map = {
        "fun": "娱乐",
        "game": "游戏",
        "sport": "体育",
    }
    if video_type:
        video_type = type_cn_map.get(video_type, video_type)

    # 假设 read_json 是你已经定义好的函数
    all_video_score_list = read_json(ALL_GOOD_VIDEO_FILE)

    # 1. 直接获取目标视频列表（比原来先全量分组更节省性能）
    if video_type:
        target_video_list = [v for v in all_video_score_list if v.get('video_type', '娱乐') == video_type]
    else:
        target_video_list = all_video_score_list

    # 2. 统一按照分数降序排序 (确保无论有无 video_type，输出的都是最高分)
    target_video_list.sort(key=lambda x: x.get('score', 0), reverse=True)

    trending_videos = []
    high_score_videos = []

    current_time = time.time()
    one_day_seconds = 4 * 60 * 60

    # 3. 遍历视频进行时间筛选和封装
    for video in target_video_list:
        # 性能优化：如果两个列表都收集满 100 个了，直接提前结束循环
        if len(trending_videos) >= 100 and len(high_score_videos) >= 100:
            break

        alive_hours = video.get('alive_hours', 0)
        score = round(video.get('score', 0) * video.get('score', 0) * video.get('score', 0) / 10, 1)
        title = video.get('title', '')
        bvid = video.get('bvid', '')
        created = video.get('created', 0)

        # 通过created和当前时间计算出视频发布的小时数
        alive_hours_new = (current_time - created) / 3600.0
        if alive_hours_new > 2 * alive_hours:
            continue

        if not (title and bvid):
            continue

        video_data = {
            "score": score,
            "title": title,
            "url": f"https://www.bilibili.com/video/{bvid}"
        }

        # 4. 判断创建时间距离现在的秒数（注：这里假设你的 created 是10位数的秒级时间戳）
        is_recent = (current_time - created) <= one_day_seconds

        if is_recent and len(trending_videos) < 100:
            trending_videos.append(video_data)
        elif not is_recent and len(high_score_videos) < 100:
            high_score_videos.append(video_data)

    return {
        "trending": trending_videos,
        "high_score": high_score_videos
    }
def get_user_video_count_lightweight(mid: int, proxies: dict = None) -> int:
    """
    新增函数：轻量级前置探测 - 获取用户公开视频总数，极低概率触发 412。
    修改点：直接写死抓包里的真实 Header 和 Cookie，无需依赖 profile 参数解耦加速。
    返回: 视频总数 (int)。如果失败返回 -1
    """
    url = "https://api.bilibili.com/x/space/navnum"
    params = {
        "mid": mid,
        "web_location": "333.1387"
    }

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36",
        "Accept": "*/*",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Origin": "https://space.bilibili.com",
        "Referer": f"https://space.bilibili.com/{mid}",
        "Cookie": "buvid3=3CEF1632-54DE-4D35-CE80-C378FB51067101804infoc; b_nut=1773895601; _uuid=3ED29ECE-C1101-4319-61031-104A4C11C37F704011infoc; home_feed_column=5; buvid_fp=723558e46b5a5d0b02b0f5bb22db0895; browser_resolution=1862-925; buvid4=0CF18BA3-8AE3-F557-86FB-7E80703A20F102641-026031912-XJY3ejH5lbSGGSHm/4GSTA%3D%3D; sid=4trggqol; CURRENT_QUALITY=0; rpdid=|(k))kmuuu)k0J\'u~~J)mlY|); bili_ticket=eyJhbGciOiJIUzI1NiIsImtpZCI6InMwMyIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3NzQxNTQ4MTIsImlhdCI6MTc3Mzg5NTU1MiwicGx0IjotMX0.hq396y0roK3Z7pEMeG9oHbEFy0Cnsj6iVoXXTetICBI; bili_ticket_expires=1774154752; CURRENT_FNVAL=2000; b_lsid=3F555719_19D047835DD"
    }

    try:
        resp = requests.get(url, params=params, headers=headers, proxies=proxies, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") == 0:
            return data.get("data", {}).get("video", -1)
        else:
            return -1
    except Exception:
        return -1


COOLDOWN_SECONDS = 4 * 3600
INTERVAL_SLEEP = 30

def run_extended_tasks():
    """封装需要额外执行的三个函数"""
    print(f"[{datetime.now()}] 触发扩展任务执行...")
    search_good_user()
    get_all_user_video_info()
    # update_uid_type()

if __name__ == "__main__":
    counter, finish_streak, last_run_time = 0, 0, 0

    while True:
        try:
            # 1. 执行核心任务并更新计数器
            is_finish = update_good_user_video()
            counter += 1
            finish_streak = (finish_streak + 1) if is_finish else 0

            # 2. 检查是否满足触发条件（阈值满足 且 时间间隔满足）
            threshold_met = (counter >= 100 or finish_streak >= 20)
            time_met = (time.time() - last_run_time >= COOLDOWN_SECONDS)

            if threshold_met and time_met:
                run_extended_tasks()
                # 重置所有状态
                counter, finish_streak, last_run_time = 0, 0, time.time()
            elif threshold_met:
                print(f"阈值已达，但冷却中...剩余 {(COOLDOWN_SECONDS - (time.time()-last_run_time))/60:.1f} 分钟")

        except Exception as e:
            traceback.print_exc()

        print(f"Wait 30s.. Counter:{counter} Streak:{finish_streak}")
        time.sleep(INTERVAL_SLEEP)
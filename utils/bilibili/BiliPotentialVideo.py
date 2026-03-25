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
from utils.proxy_utils import get_proxy

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

def get_user_videos_via_worker(mid: int,
                               worker_url: str = "https://muddy-thunder-a21b.zhuxiaohu98.workers.dev/",
                               desired_count: int = 30,
                               order: str = 'pubdate',
                               local_proxy: str = "http://127.0.0.1:7890") -> list:
    """
    纯粹的 Cloudflare Worker 转发版，专注拉取数据。
    满足要求：
    1. 代理仅在 session 级别生效，不污染全局环境变量
    2. 只有 mid 是必填参数
    3. 极致稳健，任何报错均返回空列表 []
    """
    # 【核心 3：绝对稳健】全量逻辑被 Try/Except 包裹，阻断一切崩溃可能
    try:
        headers = {
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
            "TE": "trailers",
            "Cookie": "buvid3=ADC48BBC-C48D-32BD-6E56-4BFB55D0DE9603414infoc; b_nut=1772839403; __at_once=18129603945248101006; buvid4=B47E505C-ED4B-8166-D3D2-A6A677005C3404828-026030707-tPguV4f7Z30ul7OW%2FTU71Q%3D%3D; bili_ticket=eyJhbGciOiJIUzI1NiIsImtpZCI6InMwMyIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3NzMwOTg2MDUsImlhdCI6MTc3MjgzOTM0NSwicGx0IjotMX0.ZzJwVNy04i8RjUhw9kpXI_CBY-n2vfmL8BWtEfKIgGU; bili_ticket_expires=1773098545; buvid_fp=446700a5f2f4f41f3e38bd98cabbc908; CURRENT_FNVAL=2000; sid=gqvw5gl7",
            "Referer": f"https://space.bilibili.com/{mid}/upload/video"
        }

        session = requests.Session()
        session.headers.update(headers)

        # 【核心 1：隔离代理影响】通过 Session 局部注入代理，用完即毁，绝不污染系统环境变量
        if local_proxy:
            session.proxies = {
                "http": local_proxy,
                "https": local_proxy
            }

        # ---------------- 内部工具函数 ----------------
        def get_mixin_key(orig: str) -> str:
            mixin_key_enc_tab = [
                46, 47, 18, 2, 53, 8, 23, 32, 15, 50, 10, 31, 58, 3, 45, 35, 27, 43, 5, 49,
                33, 9, 42, 19, 29, 28, 14, 39, 12, 38, 41, 13, 37, 48, 7, 16, 24, 55, 40,
                61, 26, 17, 0, 1, 60, 51, 30, 4, 22, 25, 54, 21, 56, 59, 6, 63, 57, 62, 11,
                36, 20, 34, 44, 52
            ]
            return ''.join([orig[i] for i in mixin_key_enc_tab])[:32]

        def get_wbi_keys() -> tuple:
            nav_url = "https://api.bilibili.com/x/web-interface/nav"
            resp = session.get(worker_url, params={"url": nav_url}, timeout=30)
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

        # ---------------- 主逻辑 ----------------
        img_key, sub_key = get_wbi_keys()

        collected_videos = []
        current_page = 1
        page_size = 40

        while len(collected_videos) < desired_count:
            unsigned_params = {
                'pn': current_page, 'ps': page_size, 'tid': 0, 'special_type': '',
                'order': order, 'mid': mid, 'index': 0, 'keyword': '', 'order_avoided': 'true',
                'platform': 'web',
                'web_location': '333.1387',
                'dm_img_list': '[]',
                'dm_img_str': 'V2ViR0wgMS',
                'dm_cover_img_str': 'QU5HTEUgKE5WSURJQSwgTlZJRElBIEdlRm9yY2UgR1RYIDk4MCBEaXJlY3QzRDExIHZzXzVfMCBwc181XzApLCBvciBzaW1pbGFyR29vZ2xlIEluYy4gKE5WSURJQS',
                'dm_img_inter': '{"ds":[],"wh":[5235,7405,15],"of":[491,982,491]}'
            }

            signed_params = sign_params_for_wbi(unsigned_params, img_key, sub_key)

            target_api = "https://api.bilibili.com/x/space/wbi/arc/search"

            req = requests.Request('GET', target_api, params=signed_params)
            prepared_url = req.prepare().url

            response = session.get(worker_url, params={"url": prepared_url}, timeout=30)
            response.raise_for_status()
            result = response.json()

            if result.get("code") != 0:
                print(f"❌ 接口报错: {result.get('message')}")
                break

            data = result.get("data")
            if not data: break

            new_videos = data.get('list', {}).get('vlist', [])
            if not new_videos:
                print("✅ 已经到底部，没有更多视频了。")
                break

            collected_videos.extend(new_videos)

            if len(collected_videos) >= data.get('page', {}).get('count', 0):
                print("✅ 已获取该用户所有公开视频。")
                break

            current_page += 1
            time.sleep(1)

        return collected_videos

    except Exception as e:
        print(f"⚠️ 函数执行期间发生异常，已静默拦截拦截: {e} {local_proxy} {worker_url}")
        return []



def get_user_videos_public(mid: int, desired_count: int = 30, order: str = 'pubdate', keyword: str = '',
                           use_proxy: bool = False, proxies: dict = None,
                           new_profile_list=None) -> tuple:
    """
    独立且免(登录)Cookie的B站用户视频获取函数 - 并发防风控版
    修改点：增加返回使用的 proxies 字典，方便调用层进行代理质量统计。
    """

    # 【核心优化】：身份池。将配套的 UA、Headers、Cookie 和 显卡指纹强绑定
    if new_profile_list:
        PROFILES = new_profile_list

    if use_proxy and proxies is None:
        proxies = random.choice([
    {"http": "http://viyvlyeo:lfklf4e2v9qm@31.59.20.176:6754", "https": "http://viyvlyeo:lfklf4e2v9qm@31.59.20.176:6754"},
    {"http": "http://viyvlyeo:lfklf4e2v9qm@23.95.150.145:6114", "https": "http://viyvlyeo:lfklf4e2v9qm@23.95.150.145:6114"},
    {"http": "http://viyvlyeo:lfklf4e2v9qm@198.23.239.134:6540", "https": "http://viyvlyeo:lfklf4e2v9qm@198.23.239.134:6540"},
    {"http": "http://viyvlyeo:lfklf4e2v9qm@45.38.107.97:6014", "https": "http://viyvlyeo:lfklf4e2v9qm@45.38.107.97:6014"},
    {"http": "http://viyvlyeo:lfklf4e2v9qm@107.172.163.27:6543", "https": "http://viyvlyeo:lfklf4e2v9qm@107.172.163.27:6543"},
    {"http": "http://viyvlyeo:lfklf4e2v9qm@198.105.121.200:6462", "https": "http://viyvlyeo:lfklf4e2v9qm@198.105.121.200:6462"},
    {"http": "http://viyvlyeo:lfklf4e2v9qm@64.137.96.74:6641", "https": "http://viyvlyeo:lfklf4e2v9qm@64.137.96.74:6641"},
    {"http": "http://viyvlyeo:lfklf4e2v9qm@216.10.27.159:6837", "https": "http://viyvlyeo:lfklf4e2v9qm@216.10.27.159:6837"},
    {"http": "http://viyvlyeo:lfklf4e2v9qm@142.111.67.146:5611", "https": "http://viyvlyeo:lfklf4e2v9qm@142.111.67.146:5611"},
    {"http": "http://viyvlyeo:lfklf4e2v9qm@191.96.254.138:6185", "https": "http://viyvlyeo:lfklf4e2v9qm@191.96.254.138:6185"}
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
        resp = session.get("https://api.bilibili.com/x/web-interface/nav", proxies=proxies, timeout=30)
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

    try:
        img_key, sub_key = get_wbi_keys()
    except Exception as e:
        print(f"初始化 WBI Keys 失败，可能 IP/Cookie 已被风控: {e} {proxies}")
        return [], random_index, proxies  # 修改点：将 proxies 随错误一起返回

    collected_videos = []
    current_page = 1
    page_size = 40

    while len(collected_videos) < desired_count:
        # 将动态参数与当前 profile 绑定的显卡指纹参数合并
        unsigned_params = {
            'pn': current_page, 'ps': page_size, 'tid': 0, 'special_type': '',
            'order': order, 'mid': mid, 'index': 0, 'keyword': keyword, 'order_avoided': 'true',
            **profile["dm_params"]
        }

        signed_params = sign_params_for_wbi(unsigned_params, img_key, sub_key)

        try:
            url = "https://api.bilibili.com/x/space/wbi/arc/search"
            response = session.get(url, params=signed_params, proxies=proxies, timeout=30)
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

        except Exception as e:
            print(f"查询请求发生网络或未知错误: proxies {proxies} random_index：{random_index}  {e}")
            break

    final_result = collected_videos[:desired_count]
    return final_result, random_index, proxies  # 修改点：返回使用的 proxies


def calculate_video_scores(video_list, current_timestamp, windows=(6, 12, 24, 36, 48, 100000)):
    need_filed_list = ['created', 'play', 'title', 'bvid', 'mid', 'length']
    if not video_list:
        return []

    three_days_seconds = 3 * 24 * 3600
    recent_video_count = sum(1 for v in video_list if v.get('created', 0) >= current_timestamp - three_days_seconds)
    avg_daily_videos = recent_video_count / 3.0

    for v in video_list:
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
        v['score'] = math.log(v['abs_score'] + 1) * math.log(v['abs_score'] + 1) * math.log(v['comp_score'] + 1)
        v['avg_daily_videos'] = avg_daily_videos
        v['update_time_str'] = datetime.fromtimestamp(current_timestamp).strftime("%Y-%m-%d %H:%M:%S")
        v['update_time'] = current_timestamp

    video_list.sort(key=lambda x: x['score'], reverse=True)
    return video_list


def get_user_video_count_lightweight(mid: int, proxies: dict = None) -> int:
    """
    轻量级前置探测 - 获取用户公开视频总数，极低概率触发 412。
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
        resp = requests.get(url, params=params, headers=headers, proxies=proxies, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") == 0:
            return data.get("data", {}).get("video", -1)
        else:
            return -1
    except Exception:
        return -1


def check_user_need_heavy_request(uid, exist_video_info, max_hour, new_profile_list):
    """
    【新增修改点】：提取的高可读性前置过滤逻辑。
    专门用来判断该用户是否需要进入真正的重度(WBI)拉取环节。
    返回: (is_skip: bool, skip_code: int, light_status: int, current_video_count: int, random_index: int)
    skip_code 含义: 0 (时间冷却跳过), 2 (轻量探测成功拦截跳过), -1 (不跳过，需重度拉取)
    """
    update_time = exist_video_info.get('update_time', 0)
    last_video_count = exist_video_info.get('total_video_count', 0)
    exist_video_list = exist_video_info.get('video_list', [])

    # 1. 基础时间冷却判断
    if time.time() - update_time < max_hour * 3600:
        return True, 0, 0, -1, 0  # 冷却跳过，无需再做轻量探测

    # 2. 轻量级前置探测逻辑
    random_index = random.randint(0, len(new_profile_list) - 1) if new_profile_list else 0
    current_video_count = get_user_video_count_lightweight(uid)
    light_status = -1 if current_video_count == -1 else 1

    # 如果轻量探测成功，并且之前有记录过视频数量
    if current_video_count != -1 and last_video_count > 0:
        if current_video_count <= last_video_count:
            # 核心拦截点：视频总数没涨。但需要额外判断有没有处于"爆发期"(4小时内)的视频
            latest_video_time = max([v.get('created', 0) for v in exist_video_list], default=0)

            # 只有当最新视频的发布时间距今也超过了 4 小时，才真正执行拦截跳过
            if (time.time() - latest_video_time) > 4 * 3600:
                return True, 2, light_status, current_video_count, random_index

    # 3. 确实发新视频了、或首次拉取、或轻量探测失败做兜底，允许进入重度拉取
    return False, -1, light_status, current_video_count, random_index


def process_single_user(uid, all_video_info, data_lock, max_hour=24, new_profile_list=None, proxies_list=None):
    """
    修改点：引入外置的判定逻辑，让这部分核心主流程更加直观，只处理真正需要拉取的用户。
    并将最终使用的 proxy 继续透传回调度器。
    """
    with data_lock:
        exist_video_info = all_video_info.get(str(uid), {}).copy()
        exist_video_list = exist_video_info.get('video_list', [])
        last_video_count = exist_video_info.get('total_video_count', 0)

    # ==== 提取出的高可读性判断逻辑 ====
    is_skip, skip_code, light_status, current_video_count, random_index = check_user_need_heavy_request(
        uid, exist_video_info, max_hour, new_profile_list
    )

    if is_skip:
        if skip_code == 2:  # 轻量探测成功拦截
            exist_video_info['update_time'] = int(time.time())
            exist_video_info['update_time_str'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with data_lock:
                all_video_info[str(uid)] = exist_video_info
            return 2, random_index, light_status, None
        else:  # 时间冷却跳过 (skip_code == 0)
            return 0, -1, 0, None

    # ==== 1. 先随机选择代理 ====
    proxies = None
    if proxies_list:
        proxies = random.choice(proxies_list)

    # ==== 2. 随机决定路线 (抛硬币：True代表Worker，False代表老办法) ====
    use_worker = random.choice([True, False, False])

    # ==== 3. 真正需要拉取的用户进入这里 ====
    if use_worker:
        # 【按您的绝佳思路】：直接提取字典里的 'http' 字符串传给 local_proxy，原函数完全不用改！
        proxy_str = proxies.get("http") if proxies else None
        random_value = random.random()
        if random_value > 0.5:
            proxy_str = "http://127.0.0.1:7890"

        worker_url_list = ["https://clear-emu-39.zhuxiaohu98.deno.net/",
                           "https://vercel-proxy-kappa-ruddy.vercel.app/api",
                           "https://muddy-thunder-a21b.zhuxiaohu98.workers.dev/"]
        worker_url = random.choice(worker_url_list)  # 随机选择一个 Worker URL，增加冗余和稳定性
        videos = get_user_videos_via_worker(mid=uid, desired_count=40, local_proxy=proxy_str, worker_url=worker_url)
        used_index = random_index
        used_proxy = f'{proxy_str}_{worker_url}'  # 依然如实记录这次使用了哪个代理字典
    else:
        # 走老方法
        videos, used_index, used_proxy = get_user_videos_public(
            mid=uid,
            desired_count=40,
            use_proxy=True,
            new_profile_list=new_profile_list,
            proxies=proxies
        )

    # ==== 下方逻辑一字未改，严格遵守您的要求 ====
    min_created_timestamp = 1000000000000

    if videos:
        for video_info in videos:
            created_timestamp = video_info.get('created', 0)
            if min_created_timestamp > created_timestamp:
                min_created_timestamp = created_timestamp

        exist_video_list = [v for v in exist_video_list if v.get('created', 0) < min_created_timestamp]
        exist_video_list.extend(videos)

        exist_video_list.sort(key=lambda x: x.get('created', 0), reverse=True)
        exist_video_list = exist_video_list[:72]

        exist_video_info['video_list'] = exist_video_list
        exist_video_info['update_time'] = int(time.time())
        exist_video_info['update_time_str'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if current_video_count != -1:
            exist_video_info['total_video_count'] = current_video_count
        elif last_video_count == 0:
            exist_video_info['total_video_count'] = len(exist_video_list)

        with data_lock:
            all_video_info[str(uid)] = exist_video_info
        return 1, used_index, light_status, used_proxy

    return -1, used_index, light_status, used_proxy


def filter_proxies(history_stats, proxies_list, max_count=20):
    """
    根据历史统计数据过滤代理列表，加入探索与利用机制。

    规则：
    1. 优质节点：24小时前更新的、或无记录的、或成功率 > 0 的。
    2. 淘汰节点：24小时内更新且成功率 == 0 的（进入候选列表）。
    3. 排序截取：按成功率降序排序，最多返回 max_count 个。
    4. 保底机制：如果优质节点不足 10 个，从候选列表补齐至 10 个。
    """
    valid_proxies_with_rate = []
    candidate_proxies_with_rate = []

    current_time = time.time()
    # 24小时对应的秒数
    TWENTY_FOUR_HOURS = 12 * 3600

    for proxy in proxies_list:
        p_str = proxy.get("http")
        stats = history_stats.get(p_str)

        if stats:
            last_update = stats.get('update_time', 0)
            success_rate = stats.get('success_rate', 0.2)

            # 判断逻辑：24小时前更新 OR 成功率 > 0
            if (current_time - last_update) > TWENTY_FOUR_HOURS or success_rate >= 0.5:
                # 存入元组 (代理字典, 成功率) 以便后续排序
                valid_proxies_with_rate.append((proxy, success_rate))
            else:
                # 24小时内更新，且成功率为0，属于近期明确失效节点
                candidate_proxies_with_rate.append((proxy, success_rate))
        else:
            # 没有记录的新节点，作为优质节点给它机会，默认成功率算 0 以便排序放到已知高成功率节点后面
            valid_proxies_with_rate.append((proxy, 0.5))

    # 按照成功率降序排序 (成功率高的排前面)
    valid_proxies_with_rate.sort(key=lambda x: x[1], reverse=True)

    initial_valid_count = len(valid_proxies_with_rate)
    padded_count = 0

    # 检查是否需要保底补齐 (10个)
    if initial_valid_count < 10:
        needed = 10 - initial_valid_count
        # 从候选列表(死节点)中取所需数量来补齐
        pad_list = candidate_proxies_with_rate[:needed]
        valid_proxies_with_rate.extend(pad_list)
        padded_count = len(pad_list)

    # 截取最大数量 max_count (默认15个)
    final_proxies_with_rate = valid_proxies_with_rate[:max_count]

    # 提取纯代理字典用于返回
    final_proxies = [item[0] for item in final_proxies_with_rate]

    # 打印重要信息报表
    print(
        f"[代理过滤] 输入池: {len(proxies_list)} 个 | 优质达标(>24h/无记录/成功率>0): {initial_valid_count} 个 | 触发保底补齐: {padded_count} 个 | 最终输出可用: {len(final_proxies)} 个 (上限 {max_count}) 最低成功率: {final_proxies_with_rate[-1] if final_proxies_with_rate else 'N/A'}")

    return final_proxies


def process_mid_list_concurrently(all_mid_list, all_video_info, max_workers=5, save_interval=20, max_hour=24,
                                  max_run_time=14400):
    """
    修改点：
    1. 增加 proxy_stats 字典，统计并打印每种代理的成功、失败及总量情况。
    2. 增加 max_run_time 默认 14400 秒 (4小时) 的全局超时控制。
    3. 新增将 proxy_stats 增量持久化到本地 json 的逻辑，供 get_proxy 过滤使用。
    """
    proxy_stats_file = r"W:\project\python_project\auto_video\config\proxy_stats.json"
    history_stats = read_json(proxy_stats_file)
    total_mids = len(all_mid_list)
    data_lock = threading.Lock()
    max_retries = 5
    fail_count = 200
    proxies_list = get_proxy()
    proxies_list = filter_proxies(history_stats, proxies_list)  # 根据历史统计过滤代理列表
    global_start_time = time.time()  # 新增：记录整个函数的全局开始时间
    timeout_triggered = False  # 新增：全局超时标志位

    for attempt in range(1, max_retries + 1):
        round_start_time = time.time()  # 保持原有的单轮计时用于日志打印
        success_count = 0
        fail_count = 0
        jump_count = 0
        light_skip_count = 0
        light_total_count = 0
        light_fail_count = 0
        heavy_request_count = 0
        processed_since_save = 0
        new_profile_list = BASE_PROFILES.copy()

        for i in range(1):
            new_profile = get_bilibili_fresh_profile()
            if new_profile:
                new_profile_list.append(new_profile)

        profile_stats = {}
        proxy_stats = {}  # 新增：代理追踪统计表

        print(
            f"\n--- 开始多线程处理 (第 {attempt}/{max_retries} 轮)，共提取 {total_mids} 个独立用户，设定最大并发线程: {max_workers} ---")

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_mid = {
                executor.submit(process_single_user, mid, all_video_info, data_lock, max_hour, new_profile_list,
                                proxies_list): mid
                for mid in all_mid_list
            }

            for index, future in enumerate(concurrent.futures.as_completed(future_to_mid)):
                mid = future_to_mid[future]

                # 修改点：全局超时检测与等待队列清空
                if not timeout_triggered:
                    if (time.time() - global_start_time) > max_run_time:
                        print(f"\n[超时控制] 全局运行时间已超过最大限制 ({max_run_time}秒)，正在清空等待队列...")
                        cancel_count = 0
                        for f in future_to_mid:
                            # cancel() 只会取消等待中的任务，不会打断正在执行的线程
                            if f.cancel():
                                cancel_count += 1
                        print(f"[超时控制] 成功清理 {cancel_count} 个排队中的任务，正在等待当前运行中的任务收尾...")
                        timeout_triggered = True

                try:
                    # 修改点：解包接收 4 个返回值，包含了使用的代理信息
                    result, used_index, light_status, used_proxy = future.result()

                    # 归档统计使用的 proxy
                    # 注意：只有在发生重度请求(result为 1或-1)时，才统计 proxy 使用情况
                    if result in (1, -1):
                        if used_proxy is not None:
                            # 提取字典中的 http 字段用于记录，或将其转为字符串
                            if isinstance(used_proxy, dict) and 'http' in used_proxy:
                                proxy_str = used_proxy['http']
                            else:
                                proxy_str = str(used_proxy)
                        else:
                            proxy_str = "None(直连)"

                        if proxy_str not in proxy_stats:
                            proxy_stats[proxy_str] = {'success': 0, 'failed': 0, 'total': 0}

                        proxy_stats[proxy_str]['total'] += 1
                        if result == 1:
                            proxy_stats[proxy_str]['success'] += 1
                        else:
                            proxy_stats[proxy_str]['failed'] += 1

                    # 归档统计使用的 profile
                    if used_index != -1:
                        if used_index not in profile_stats:
                            profile_stats[used_index] = {'used': 0, 'failed': 0}
                        profile_stats[used_index]['used'] += 1

                    # 统计轻量接口的数据
                    if light_status != 0:
                        light_total_count += 1
                        if light_status == -1:
                            light_fail_count += 1

                    # 统计重度接口及拦截的数据
                    if result == 1:
                        success_count += 1
                        heavy_request_count += 1
                        processed_since_save += 1
                    elif result == -1:
                        fail_count += 1
                        heavy_request_count += 1
                        if used_index != -1:
                            profile_stats[used_index]['failed'] += 1
                    elif result == 2:
                        light_skip_count += 1
                    else:  # result == 0
                        jump_count += 1

                    if index % 100 == 0 or index == total_mids - 1:
                        print(
                            f"\n正在处理 mid: {mid} 进度: {index + 1}/{total_mids} | 冷却跳过: {jump_count} | 轻量探测: {light_total_count}(失败{light_fail_count} 拦截{light_skip_count}) | 实际重度请求: {heavy_request_count}(成功{success_count}/失败{fail_count}) | 当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

                    if processed_since_save >= save_interval:
                        with data_lock:
                            save_json(ALL_VIDEO_FILE, all_video_info)
                        print(f"\n>>>> 触发批量定时保存：最新 {processed_since_save} 个用户数据已落盘 <<<<\n")
                        processed_since_save = 0

                except concurrent.futures.CancelledError:
                    # 被超时机制 cancel 的任务在获取 result() 时会触发此异常，直接跳过即可
                    continue
                except Exception as e:
                    traceback.print_exc()
                    print(f"处理用户 mid: {mid} 发生异常: {e}")
                    fail_count += 1
                    heavy_request_count += 1

        if processed_since_save > 0:
            with data_lock:
                save_json(ALL_VIDEO_FILE, all_video_info)
            print(f"\n>>>> 触发最终扫尾保存：剩余的 {processed_since_save} 个新用户数据已落盘 <<<<\n")

        print(f"\n--- 第 {attempt} 轮多线程处理完成！总耗时: {time.time() - round_start_time:.2f} 秒。 ---")

        print(">>> 📊 本轮 PROFILES (Index) 详细统计 <<<")
        if profile_stats:
            for idx in sorted(profile_stats.keys()):
                stats = profile_stats[idx]
                fail_rate = (stats['failed'] / stats['used']) * 100 if stats['used'] > 0 else 0
                print(
                    f"  Profile Index [{idx}]: 尝试次数: {stats['used']:<4} | 失败次数: {stats['failed']:<4} | 失败率: {fail_rate:.2f}%")
        else:
            print("  本轮没有使用任何 Profile (可能是用户数据都在未过期跳过范围内)。")
        print(">>> -------------------------------------- <<<")

        # 修改点：在此处新增打印本轮 IP代理（Proxies） 的详细统计报表 (已增加成功率降序排序逻辑)
        print(">>> 🌐 本轮 IP代理 (Proxies) 详细统计 <<<")
        if proxy_stats:
            # 按照成功率 (success/total) 降序排序
            sorted_proxy_stats = sorted(
                proxy_stats.items(),
                key=lambda x: (x[1]['success'] / x[1]['total']) if x[1]['total'] > 0 else 0,
                reverse=True
            )
            for p_str, stats in sorted_proxy_stats:
                success_rate = (stats['success'] / stats['total']) * 100 if stats['total'] > 0 else 0
                fail_rate = (stats['failed'] / stats['total']) * 100 if stats['total'] > 0 else 0
                print(
                    f"  代理 [{p_str}]: 总调度次数: {stats['total']:<4} | 成功: {stats['success']:<4} | 失败: {stats['failed']:<4} | 成功率: {success_rate:>6.2f}% | 失败率: {fail_rate:>6.2f}%")
        else:
            print("  本轮未触发需要真正拉取的重度请求，无代理调度记录。")
        print(
            f"统计汇总 -> 时间冷却跳过: {jump_count} | 轻量探测发起: {light_total_count} (失败: {light_fail_count}) -> 成功拦截重度请求: {light_skip_count} | 实际发起WBI重度请求: {heavy_request_count} (成功: {success_count}, 失败: {fail_count})\n")

        print(">>> -------------------------------------- <<<\n")

        # ================== 新增：代理统计持久化落盘逻辑 ==================
        if proxy_stats:
            history_stats = read_json(proxy_stats_file)

            current_timestamp = time.time()
            for p_str, stats in proxy_stats.items():
                if p_str not in history_stats:
                    history_stats[p_str] = {'total': 0, 'success': 0, 'failed': 0}

                # 累加历史数据
                h_stats = history_stats[p_str]
                if h_stats['total'] > 100:
                    h_stats['total'] = 0
                    h_stats['success'] = 0
                    h_stats['failed'] = 0

                h_stats['total'] += stats['total']
                h_stats['success'] += stats['success']
                h_stats['failed'] += stats['failed']

                # 更新时间和成功/失败率
                h_stats['update_time'] = current_timestamp
                h_stats['success_rate'] = h_stats['success'] / h_stats['total'] if h_stats['total'] > 0 else 0
                h_stats['fail_rate'] = h_stats['failed'] / h_stats['total'] if h_stats['total'] > 0 else 0
            save_json(proxy_stats_file, history_stats)


        # 修改点：判断全局超时标志
        if timeout_triggered:
            print(f"[超时退出] 全局运行时间已超过最大限制({max_run_time}秒)，自动结束本阶段执行，不再重试。")
            break

        if fail_count <= 200:
            print(f"失败数量({fail_count})在可接受范围内(<=200)，无需重试，正常结束本阶段处理。")
            break
        else:
            if attempt < max_retries:
                print(f"注意：检测到 fail_count({fail_count}) > 200，准备触发原路重试机制，进行第 {attempt + 1} 轮执行...")
                time.sleep(10)
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
    need_filed_list = [ 'mid', 'aid', 'bvid', 'title', 'description', 'play', 'pubdate']

    for v in videos:
        keys_to_del = [k for k in v if k not in need_filed_list]
        for k in keys_to_del:
            del v[k]

    print(
        f"搜索标签 {tag} 得到 {len(videos)} 个相关视频。 耗时 {time.time() - start_time:.2f} 秒。 当前时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if videos:
        exist_video_info['video_info_list'] = videos
        exist_video_info['update_time'] = int(time.time())
        all_user_info[str(tag)] = exist_video_info
        save_json(ALL_USER_FILE, all_user_info)
    else:
        print(f"搜索标签 {tag} 没有得到数据，可能被风控了，等待10秒后再试...")
        time.sleep(10)


def load_pure_user_info():
    all_user_info = read_json(ALL_USER_FILE)
    need_filed_list = [ 'mid', 'aid', 'bvid', 'title', 'description', 'play', 'pubdate']

    for tag, video_info in all_user_info.items():
        video_info_list = video_info.get('video_info_list', [])
        if not video_info_list:
            continue

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
    need_filed_list = ['created', 'play', 'title', 'bvid', 'mid', 'length']

    for uid, exist_video_info in all_video_info.items():
        exist_video_list = exist_video_info.get('video_list', [])
        if not exist_video_list:
            continue

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

    # 1. 先计算并打印仅来自于 tag 搜索的去重个数
    tag_mids_set = set(all_mid_list)
    tag_count = len(tag_mids_set)
    print(f"从所有标签的视频信息中提取到 {tag_count} 个唯一用户 mid。")

    # 必须加载 all_video_info，因为后面的并发函数需要用它来比对和保存数据
    all_video_info = load_pure_video_info()

    # 获取当前小时数 (0-23)
    current_hour = datetime.now().hour

    # 判断是否在凌晨 0 点到 6 点之间 (即 0:00 到 5:59)
    if 0 <= current_hour < 6:
        # 2. 获取历史库中的所有 uid 集合
        history_mids_set = set(int(uid) for uid in all_video_info.keys())

        # 3. 计算历史库排除了 tag 已有 uid 后，实际额外增加的个数
        new_added_count = len(history_mids_set - tag_mids_set)

        # 4. 合并两路数据，得到最终的去重列表
        all_mid_list = list(tag_mids_set | history_mids_set)

        print(f"当前时间 {current_hour} 点，满足凌晨更新条件。从全量历史库又增加了 {new_added_count} 个唯一用户 mid，本次总计待处理 {len(all_mid_list)} 个。")
    else:
        # 不在凌晨时段，只处理刚才搜出来的 tag 用户
        all_mid_list = list(tag_mids_set)
        print(f"当前时间 {current_hour} 点，未在凌晨(0-6点)时段，跳过全量历史库追加。本次总计待处理 {len(all_mid_list)} 个。")

    process_mid_list_concurrently(all_mid_list, all_video_info, max_workers=20, save_interval=1000)

def gen_uid_type_llm(uid_info_list):
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
                time.sleep(retry_delay)
            else:
                print(f"达到最大重试次数，失败. {log_pre}")
                return None


def update_uid_type():
    start_time = time.time()
    all_video_score_list, all_video_info = get_sorted_high_score_videos()

    all_user_type_info = read_json(ALL_USER_TYPE_MAP_FILE)
    if not isinstance(all_user_type_info, dict):
        all_user_type_info = {}

    all_video_info = load_pure_video_info()
    unique_uids = set(v['mid'] for v in all_video_score_list)

    current_time = time.time()
    uids_to_update = []

    for uid in unique_uids:
        uid_str = str(uid)
        if uid_str not in all_user_type_info:
            uids_to_update.append(uid)
        else:
            user_data = all_user_type_info[uid_str]
            if isinstance(user_data, dict):
                last_update = user_data.get('update_time', 0)
                if current_time - last_update > 24 * 3600:
                    uids_to_update.append(uid)
            else:
                uids_to_update.append(uid)

    if not uids_to_update:
        print("没有检测到需要更新类型的新用户或过期用户。")
        return

    print(f"检测到 {len(uids_to_update)} 个用户需要更新类型 (新增或已超过24小时)...")

    uid_latest_titles_dict = {}
    for uid in uids_to_update:
        uid_str = str(uid)
        if uid_str in all_video_info:
            video_list = all_video_info[uid_str].get('video_list', [])
            sorted_videos = sorted(video_list, key=lambda x: x.get('created', 0), reverse=True)
            latest_5_titles = [v.get('title', '') for v in sorted_videos[:5]]
            uid_latest_titles_dict[uid_str] = latest_5_titles

    batch_size = 50
    batch_list = []
    current_batch = {}

    for uid, titles in uid_latest_titles_dict.items():
        current_batch[uid] = titles
        if len(current_batch) == batch_size:
            batch_list.append(current_batch)
            current_batch = {}

    if current_batch:
        batch_list.append(current_batch)

    for index, batch in enumerate(batch_list):
        batch_start_time = time.time()
        try:
            print(
                f"\n\n正在处理第 {index + 1} 批次，共 {len(batch_list)} 批次，当前批次包含 {len(batch)} 个用户... 当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            type_result_dict = gen_uid_type_llm(batch)

            if type_result_dict:
                for uid, v_type in type_result_dict.items():
                    if v_type in ["体育", "游戏", "娱乐", "军事"]:
                        all_user_type_info[str(uid)] = {
                            "type": v_type,
                            "update_time": int(time.time())
                        }

                save_json(ALL_USER_TYPE_MAP_FILE, all_user_type_info)
                print(
                    f"Batch {index + 1}/{len(batch_list)} 更新完毕并保存，本批次处理 {len(type_result_dict)} 个用户。 耗时 {time.time() - batch_start_time:.2f} 秒。")
            else:
                print(f"Batch {index + 1}/{len(batch_list)} gen_uid_type_llm 返回为空，跳过保存。")

        except Exception as e:
            print(f"Batch {index + 1}/{len(batch_list)} 调用 LLM 或保存时发生异常: {e}")
            traceback.print_exc()

    print(
        f"所有批次处理完成！总耗时: {time.time() - start_time:.2f} 秒。总共更新了 {len(uids_to_update)} 个用户的类型。 批次数量: {len(batch_list)}。")


def get_sorted_high_score_videos(max_hour=24):
    all_video_info = load_pure_video_info()
    all_user_type_info = read_json(ALL_USER_TYPE_MAP_FILE)

    stats = {
        "total_users": len(all_video_info),
        "filter_invalid_type": 0,
        "filter_update_timeout": 0,
        "filter_no_recent_video": 0,
        "processed_users": 0
    }

    all_video_score_list = []
    current_now = time.time()
    total_count = 0
    for uid, exist_video_info in all_video_info.items():
        exist_video_list = exist_video_info.get('video_list', [])

        total_count += len(exist_video_list)
        user_type_data = all_user_type_info.get(str(uid))
        if isinstance(user_type_data, dict):
            video_type = user_type_data.get('type', "未知")
        elif isinstance(user_type_data, str):
            video_type = user_type_data
        else:
            video_type = "未知"

        if video_type not in ['娱乐', '游戏', '体育']:
            stats["filter_invalid_type"] += 1
            continue

        update_time = exist_video_info.get('update_time', 0)

        if current_now - update_time > max_hour * 3600:
            stats["filter_update_timeout"] += 1
            continue

        latest_created_time = max((v.get('created', 0) for v in exist_video_list), default=0)

        if current_now - latest_created_time > 24 * 3600:
            stats["filter_no_recent_video"] += 1
            continue

        video_score_list = calculate_video_scores(exist_video_list, current_timestamp=update_time)

        for v in video_score_list:
            v['video_type'] = video_type

        stats["processed_users"] += 1
        all_video_score_list.extend(video_score_list)

    all_video_score_list.sort(key=lambda x: x.get('score', 0), reverse=True)

    print("-" * 30)
    print(f"【处理统计报告】")
    print(f"1. 总用户数: {stats['total_users']} 过滤前总视频数: {total_count}")
    print(f"2. 过滤-题材不符 (非娱乐/游戏/体育): {stats['filter_invalid_type']}")
    print(f"3. 过滤-更新超时 (>{max_hour}h): {stats['filter_update_timeout']}")
    print(f"4. 过滤-24h内无新视频: {stats['filter_no_recent_video']}")
    print(f"5. 最终处理用户数: {stats['processed_users']}")
    print(f"6. 最终生成视频分数记录数: {len(all_video_score_list)}")
    print("-" * 30)

    all_video_score_list = [v for v in all_video_score_list if v.get('avg_daily_videos', 0) > 1.0]
    all_video_score_list = [v for v in all_video_score_list if v.get('duration', 0) < 600.0]
    all_video_score_list = [v for v in all_video_score_list if v.get('alive_hours', 0) < 72]

    all_video_score_list.sort(key=lambda x: x['score'], reverse=True)

    good_video_count = 1000
    type_count_map = {}
    pure_all_video_score_list = []

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
    all_video_score_list, all_video_info = get_sorted_high_score_videos()

    unique_uids = set(v['mid'] for v in all_video_score_list)
    save_json(ALL_GOOD_USER_FILE, list(unique_uids))
    print(
        f"筛选完成，当前共有 {len(all_video_score_list)} 个符合条件的高分视频。 来源于 {len(unique_uids)} 个不同的用户。")

    max_hour = 1
    if not (5 <= datetime.now().hour < 24):
        max_hour = 2

    is_finish = process_mid_list_concurrently(unique_uids, all_video_info, max_workers=20, save_interval=1000,
                                              max_hour=max_hour)

    return is_finish


def get_good_video(video_type=None):
    type_cn_map = {
        "fun": "娱乐",
        "game": "游戏",
        "sport": "体育",
    }
    if video_type:
        video_type = type_cn_map.get(video_type, video_type)

    all_video_score_list = read_json(ALL_GOOD_VIDEO_FILE)

    if video_type:
        target_video_list = [v for v in all_video_score_list if v.get('video_type', '娱乐') == video_type]
    else:
        target_video_list = all_video_score_list

    target_video_list.sort(key=lambda x: x.get('score', 0), reverse=True)

    trending_videos = []
    high_score_videos = []

    current_time = time.time()
    one_day_seconds = 4 * 60 * 60

    for video in target_video_list:
        if len(trending_videos) >= 100 and len(high_score_videos) >= 100:
            break

        alive_hours = video.get('alive_hours', 0)
        score = round(video.get('score', 0) * video.get('score', 0) * video.get('score', 0) / 10, 1)
        title = video.get('title', '')
        bvid = video.get('bvid', '')
        created = video.get('created', 0)

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

        is_recent = (current_time - created) <= one_day_seconds

        if is_recent and len(trending_videos) < 100:
            trending_videos.append(video_data)
        elif not is_recent and len(high_score_videos) < 100:
            high_score_videos.append(video_data)

    return {
        "trending": trending_videos,
        "high_score": high_score_videos
    }


COOLDOWN_SECONDS = 4 * 3600
INTERVAL_SLEEP = 30


def run_extended_tasks():
    print(f"[{datetime.now()}] 触发扩展任务执行...")
    search_good_user()
    get_all_user_video_info()


if __name__ == "__main__":
    counter, finish_streak, last_run_time = 0, 0, 0
    # is_finish = update_good_user_video()
    # get_all_user_video_info()
    while True:
        try:
            is_finish = update_good_user_video()
            counter += 1
            finish_streak = (finish_streak + 1) if is_finish else 0

            threshold_met = (counter >= 100 or finish_streak >= 20)
            time_met = (time.time() - last_run_time >= COOLDOWN_SECONDS)

            if threshold_met and time_met:
                run_extended_tasks()
                counter, finish_streak, last_run_time = 0, 0, time.time()
            elif threshold_met:
                print(f"阈值已达，但冷却中...剩余 {(COOLDOWN_SECONDS - (time.time() - last_run_time)) / 60:.1f} 分钟")

        except Exception as e:
            traceback.print_exc()

        print(f"Wait 30s.. Counter:{counter} Streak:{finish_streak}")
        time.sleep(INTERVAL_SLEEP)
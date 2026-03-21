# multiprocess_runner.py
import concurrent.futures
import multiprocessing
from typing import Dict, Any
# 先: pip install playwright
# 然后: playwright install
# 推荐: 本地有 Chrome/Edge 安装，或者用 Playwright 的 channel="chrome"
from urllib.parse import urlparse # <--- 引入 urlparse

import time
import random
from typing import List, Dict, Optional
from playwright.sync_api import sync_playwright, TimeoutError, Error

from utils.common_utils import init_config, read_json, save_json


def _parse_cookie_string(cookie_string: str) -> Dict[str, str]:
    cookies = {}
    for part in cookie_string.split(';'):
        part = part.strip()
        if not part:
            continue
        if '=' not in part:
            continue
        k, v = part.split('=', 1)
        cookies[k.strip()] = v.strip()
    return cookies

def visit_bilibili_with_cookies_fix(
    cookie_string: str,
    video_url: str = 'https://www.bilibili.com/video/BV16Sa3zhEPe/',
    headless: bool = False,                # 默认不开 headless，尽量像真实浏览器
    watch_seconds: int = 60,
    use_system_chrome: bool = True,        # 尝试使用系统 Chrome（更可靠）
    chrome_executable_path: Optional[str] = None,  # 如果 channel 不可用，可传 chrome 可执行路径
):
    """
    改进版：尽量解决 '浏览器不支持 HTML5 播放器' 的提示。

    参数:
      cookie_string: "k1=v1; k2=v2; ..." 的 cookie 字符串（最好包含 SESSDATA）
      video_url: 目标视频
      headless: 是否无头（建议 False）
      watch_seconds: 最多观看多少秒
      use_system_chrome: 是否尝试用系统 Chrome（优先）
      chrome_executable_path: 可选，显式指定 Chrome 可执行文件路径（例如 C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe）
    返回:
      dict 运行结果
    """
    cookies_dict = _parse_cookie_string(cookie_string)
    result = {
        "likely_logged_in_from_cookie": 'SESSDATA' in cookies_dict or 'bili_jct' in cookies_dict,
        "page_video_found": False,
        "played_video": False,
        "page_login_status_guess": None,
        "error": None,
    }

    user_agent = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/116.0.0.0 Safari/537.36"
    )

    # 转换 cookie 到 Playwright 格式
    pk_cookies: List[dict] = []
    for k, v in cookies_dict.items():
        pk_cookies.append({
            "name": k,
            "value": v,
            "domain": ".bilibili.com",
            "path": "/",
        })

    p = None
    browser = None
    try:
        p = sync_playwright().start()

        launch_kwargs = dict(
            headless=headless,
            args=[
                "--no-sandbox",
                "--disable-blink-features=AutomationControlled",
                "--autoplay-policy=no-user-gesture-required",  # 允许自动播放
                "--disable-dev-shm-usage",
                "--disable-infobars",
                # 可按需加代理参数等
            ],
        )

        # 尝试优先用系统 Chrome（更可能支持 H264 等 codec）
        try:
            if use_system_chrome:
                # 尝试通过 channel（如果 Playwright 能找到 Chrome）
                browser = p.chromium.launch(channel="chrome", **launch_kwargs)
            else:
                if chrome_executable_path:
                    browser = p.chromium.launch(executable_path=chrome_executable_path, **launch_kwargs)
                else:
                    browser = p.chromium.launch(**launch_kwargs)
        except Exception as e_channel:
            # 如果 channel 不可用，尝试用显式可执行路径（若提供）
            if chrome_executable_path:
                browser = p.chromium.launch(executable_path=chrome_executable_path, **launch_kwargs)
            else:
                # 回退到默认 Playwright Chromium（但仍尽可能伪装）
                browser = p.chromium.launch(**launch_kwargs)

        context = browser.new_context(
            user_agent=user_agent,
            locale="zh-CN",
            timezone_id="Asia/Shanghai",
            viewport={"width": 1366, "height": 768},
            java_script_enabled=True,
        )

        # 注入 cookies
        if pk_cookies:
            context.add_cookies(pk_cookies)

        page = context.new_page()

        # 尽量隐藏 webdriver 指纹
        try:
            page.add_init_script("""() => {
                // Hide webdriver flag
                Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                Object.defineProperty(navigator, 'languages', {get: () => ['zh-CN','zh']});
                Object.defineProperty(navigator, 'plugins', {get: () => [1,2,3,4,5]});
                // Mock chrome object
                window.chrome = { runtime: {} };
            }""")
        except Exception:
            pass

        # 访问页面
        page.goto(video_url, wait_until="networkidle", timeout=45000)

        # 检查页面是否展示“不支持 HTML5”的提示（通过页面文本）
        try:
            # 如果页面里出现类似提示，尝试点击“切换为 HTML5 播放器”或调整 UA/重载
            if page.locator("text=不支持 HTML5").count() > 0 or page.locator("text=HTML5 播放器").count() > 0:
                result["error"] = "页面报告不支持 HTML5 播放器（检测到提示文本）。已尝试自动修复策略。"
                # 尝试寻找并点击切换按钮（若存在）
                possible_buttons = [
                    "text=切换为 HTML5 播放器",
                    "text=使用 HTML5 播放器",
                    "text=我知道了",
                ]
                for b in possible_buttons:
                    try:
                        if page.locator(b).count() > 0:
                            page.locator(b).first.click(timeout=3000)
                            time.sleep(1)
                    except Exception:
                        pass
                # 重新导航一次（让播放器重试）
                page.reload(wait_until="networkidle", timeout=30000)
        except Exception:
            pass

        # 等待 video 元素 或 帧内播放器
        video_el = None
        try:
            video_el = page.wait_for_selector("video", timeout=15000)
            result["page_video_found"] = video_el is not None
        except TimeoutError:
            result["page_video_found"] = False

        # 尝试播放
        played = False
        if result["page_video_found"]:
            try:
                played = page.evaluate("""() => {
                    const v = document.querySelector('video');
                    if (!v) return false;
                    try { v.muted = false; } catch(e){}
                    try { v.playbackRate = 1.0; } catch(e){}
                    return v.play().then(()=>true).catch(()=>false);
                }""")
            except Exception:
                played = False

        # 尝试通过点击播放按钮（备用）
        if not played:
            try:
                # 一些选择器在播放器覆盖层上
                click_selectors = [
                    ".bpx-player-ctrl .bpx-player-ctrl-play",
                    ".bilibili-player-video-btn",
                    ".play-btn",
                    "button[aria-label='播放']",
                    ".player-controller .play",
                ]
                for sel in click_selectors:
                    try:
                        loc = page.locator(sel)
                        if loc.count() > 0:
                            loc.first.click(timeout=3000)
                            time.sleep(0.8)
                            played = True
                            break
                    except Exception:
                        continue
            except Exception:
                pass

        result["played_video"] = bool(played)

        # 做简单的人类行为模拟（滚动、移动鼠标）
        start = time.time()
        elapsed = 0
        while elapsed < watch_seconds:
            time.sleep(random.uniform(1.0, 2.5))
            try:
                page.evaluate(
                    """() => {
                        const h = document.documentElement.scrollHeight || document.body.scrollHeight;
                        const y = Math.floor(Math.random() * (h / 3)) + 50;
                        window.scrollTo({ top: y, behavior: 'smooth' });
                    }"""
                )
            except Exception:
                pass

            try:
                v = page.query_selector("video")
                if v:
                    box = v.bounding_box()
                    if box:
                        x = box["x"] + random.uniform(10, max(10, box["width"] - 10))
                        y = box["y"] + random.uniform(10, max(10, box["height"] - 10))
                        page.mouse.move(x, y, steps=random.randint(3, 12))
            except Exception:
                # 随机移动
                page.mouse.move(random.uniform(100, 1200), random.uniform(100, 600), steps=5)

            elapsed = time.time() - start

        # 最后抓取一些信息
        try:
            result["final_url"] = page.url
            result["page_title"] = page.title()
            # 再次简单判断登录状态
            try:
                result["page_login_status_guess"] = "maybe_not_logged_in" if page.locator("text=登录").count() > 0 else "maybe_logged_in"
            except Exception:
                result["page_login_status_guess"] = "unknown"
        except Exception:
            pass

    except Exception as e:
        result["error"] = str(e)
    finally:
        try:
            if browser:
                browser.close()
        except Exception:
            pass
        try:
            if p:
                p.stop()
        except Exception:
            pass

    return result


def refresh_page_periodically(
        url: str,
        watch_seconds: int = 60,
        cookie_string: Optional[str] = None,
        headless: bool = False,
        use_system_chrome: bool = True,
        chrome_executable_path: Optional[str] = None,
):
    """
    使用 Playwright 访问指定页面，并在指定时间内，每隔5秒刷新一次。
    (已修复刷新卡住的问题)
    """
    p = None
    browser = None
    try:
        p = sync_playwright().start()

        launch_kwargs = dict(
            headless=headless,
            args=[
                "--no-sandbox",
                "--disable-blink-features=AutomationControlled",
                "--autoplay-policy=no-user-gesture-required",
                "--disable-dev-shm-usage",
                "--disable-infobars",
            ],
        )

        try:
            if use_system_chrome:
                browser = p.chromium.launch(channel="chrome", **launch_kwargs)
            elif chrome_executable_path:
                browser = p.chromium.launch(executable_path=chrome_executable_path, **launch_kwargs)
            else:
                browser = p.chromium.launch(**launch_kwargs)
        except Exception:
            browser = p.chromium.launch(**launch_kwargs)

        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/116.0.0.0 Safari/537.36"
            ),
            locale="zh-CN",
            timezone_id="Asia/Shanghai",
            viewport={"width": 1366, "height": 768},
            java_script_enabled=True,
        )

        if cookie_string:
            cookies_dict = _parse_cookie_string(cookie_string)
            if cookies_dict:
                parsed_url = urlparse(url)
                hostname_parts = parsed_url.hostname.split('.')
                cookie_domain = f".{'.'.join(hostname_parts[-2:])}" if len(hostname_parts) > 1 else parsed_url.hostname

                pk_cookies: List[dict] = []
                for k, v in cookies_dict.items():
                    pk_cookies.append({
                        "name": k, "value": v, "domain": cookie_domain, "path": "/"
                    })
                context.add_cookies(pk_cookies)

        page = context.new_page()

        page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
        """)

        # --- 修改点 1：首次访问页面时，使用 'domcontentloaded' ---
        print(f"正在访问页面: {url}")
        page.goto(url, wait_until="domcontentloaded", timeout=45000)  # <--- 主要修改点
        print("页面访问成功，开始刷新循环...")

        start_time = time.time()
        while time.time() - start_time < watch_seconds:
            elapsed_time = time.time() - start_time
            remaining_time = watch_seconds - elapsed_time

            sleep_duration = min(10.0, remaining_time)
            if sleep_duration <= 0:
                break

            print(f"等待 {sleep_duration:.1f} 秒后刷新... (总进度: {int(elapsed_time)}/{watch_seconds} 秒)")
            time.sleep(sleep_duration)

            # --- 修改点 2：刷新页面时，同样使用 'domcontentloaded' ---
            try:
                print("正在刷新页面...")
                page.reload(wait_until="domcontentloaded", timeout=30000)  # <--- 主要修改点
                print("刷新完成。")
            except Exception as e:
                print(f"刷新失败，尝试重新导航: {e}")
                try:
                    # --- 修改点 3：备用导航方案也使用 'domcontentloaded' ---
                    page.goto(url, wait_until="domcontentloaded", timeout=30000)  # <--- 主要修改点
                except Exception as e2:
                    print(f"重新导航也失败了: {e2}")
                    break

        print(f"已达到指定的 {watch_seconds} 秒，任务结束。")

    except Exception as e:
        print(f"在执行过程中发生错误: {e}")
        raise
    finally:
        if browser:
            browser.close()
        if p:
            p.stop()
        print("浏览器和Playwright实例已关闭。")

def _worker_call(cookie_string: str, video_url: str, headless: bool, watch_seconds: int) -> Dict[str, Any]:
    """在子进程中运行的函数（必须为顶层函数以便 pickling）。"""
    try:
        res = refresh_page_periodically(
            cookie_string=cookie_string,
            url=video_url,
            headless=headless,
            watch_seconds=watch_seconds
        )
        # 不打印完整 cookie，返回一个简短提示
        return res
    except Exception as e:
        return {"error": str(e)}

import re
import time
import urllib.parse
import requests
from hashlib import md5

from utils.common_utils import init_config


def get_user_dynamics(host_mid: int, cookie_str: str, desire_count: int = 30) -> list:
    """
    获取B站用户空间动态列表 (支持 Wbi 签名与游标翻页)

    :param host_mid: 目标用户的 UID
    :param cookie_str: 你的 Bilibili Cookie 字符串
    :param desire_count: 期望获取的动态数量
    :return: 动态列表 (包含动态的各种元数据和内容)
    """

    session = requests.Session()
    # 设置请求头，加入你抓包中的 User-Agent、Referer 和 Cookie
    session.headers.update({
        'accept': '*/*',
        'accept-language': 'zh-CN',
        'cookie': cookie_str,
        'origin': 'https://space.bilibili.com',
        'referer': f'https://space.bilibili.com/{host_mid}/dynamic',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36'
    })

    # --- Wbi 签名核心算法 (复用并适配动态接口) ---
    def get_mixin_key(orig: str) -> str:
        mixin_key_enc_tab = [
            46, 47, 18, 2, 53, 8, 23, 32, 15, 50, 10, 31, 58, 3, 45, 35, 27, 43, 5, 49,
            33, 9, 42, 19, 29, 28, 14, 39, 12, 38, 41, 13, 37, 48, 7, 16, 24, 55, 40,
            61, 26, 17, 0, 1, 60, 51, 30, 4, 22, 25, 54, 21, 56, 59, 6, 63, 57, 62, 11,
            36, 20, 34, 44, 52
        ]
        return ''.join([orig[i] for i in mixin_key_enc_tab])[:32]

    def get_wbi_keys() -> tuple:
        # 获取 Wbi 签名的盐值，依赖你的 Cookie 状态
        resp = session.get("https://api.bilibili.com/x/web-interface/nav", timeout=10)
        resp.raise_for_status()
        wbi_img = resp.json().get('data', {}).get('wbi_img', {})
        return (
            wbi_img.get('img_url', '').rsplit('/', 1)[1].split('.')[0],
            wbi_img.get('sub_url', '').rsplit('/', 1)[1].split('.')[0]
        )

    def sign_params_for_wbi(params: dict, img_key: str, sub_key: str) -> dict:
        mixin_key = get_mixin_key(img_key + sub_key)
        params['wts'] = round(time.time())
        # 对参数按 Key 的字母顺序进行排序
        sorted_params = dict(sorted(params.items()))

        encoded_parts = []
        for k, v in sorted_params.items():
            # 过滤掉特定的特殊字符
            filtered_value = ''.join(filter(lambda chr: chr not in "!'()*", str(v)))
            encoded_parts.append(f"{k}={urllib.parse.quote(filtered_value, safe='')}")

        query = '&'.join(encoded_parts)
        params['w_rid'] = md5((query + mixin_key).encode()).hexdigest()
        return params

    # 1. 获取 Wbi Keys
    try:
        img_key, sub_key = get_wbi_keys()
    except Exception as e:
        print(f"初始化 WBI Keys 失败，Cookie 可能失效或网络异常: {e}")
        return []

    collected_dynamics = []
    offset = ""  # 初始页面的 offset 必须为空字符串

    # 2. 构造抓包提取出的设备指纹等静态参数
    # Wbi 签名会严格校验所有传输的参数，因此必须保留
    base_params = {
        'timezone_offset': '-480',
        'platform': 'web',
        'features': 'itemOpusStyle,listOnlyfans,opusBigCover,onlyfansVote,forwardListHidden,decorationCard,commentsNewVersion,onlyfansAssetsV2,ugcDelete,onlyfansQaCard,avatarAutoTheme,sunflowerStyle,cardsEnhance,eva3CardOpus,eva3CardVideo,eva3CardComment,eva3CardUser',
        'web_location': '333.1387',
        'dm_img_list': '[{"x":1386,"y":767,"z":0,"timestamp":1,"k":94,"type":0},{"x":1922,"y":936,"z":73,"timestamp":43,"k":113,"type":0},{"x":2020,"y":887,"z":14,"timestamp":149,"k":89,"type":0}]',
        'dm_img_str': 'V2ViR0wgMS4wIChPcGVuR0wgRVMgMi4wIENocm9taXVtKQ',
        'dm_cover_img_str': 'QU5HTEUgKE5WSURJQSwgTlZJRElBIEdlRm9yY2UgUlRYIDIwODAgVGkgKDB4MDAwMDFFMDcpIERpcmVjdDNEMTEgdnNfNV8wIHBzXzVfMCwgRDNEMTEpR29vZ2xlIEluYy4gKE5WSURJQS',
        'dm_img_inter': '{"ds":[{"t":0,"c":"bnByb2dyZXNzLWJ1c3","p":[6,2,2],"s":[239,6090,2286]}],"wh":[4436,4752,56],"of":[76,152,76]}',
        'x-bili-device-req-json': '{"platform":"web","device":"pc","spmid":"333.1387"}'
    }

    # 3. 循环拉取动态
    while len(collected_dynamics) < desire_count:
        # 动态组装当前页的请求参数
        unsigned_params = {
            'host_mid': host_mid,
            'offset': offset,
            **base_params
        }

        # 签名运算生成 w_rid 和 wts
        signed_params = sign_params_for_wbi(unsigned_params, img_key, sub_key)

        try:
            url = "https://api.bilibili.com/x/polymer/web-dynamic/v1/feed/space"
            response = session.get(url, params=signed_params, timeout=15)
            response.raise_for_status()
            result = response.json()

            if result.get("code") != 0:
                print(f"UID: {host_mid} 接口报错，错误码: {result.get('code')}, 信息: {result.get('message')}")
                break

            data = result.get("data", {})
            if not data:
                break

            # 提取本页动态列表
            items = data.get('items', [])
            if not items:
                print(f"UID: {host_mid} 动态到底了。")
                break

            collected_dynamics.extend(items)

            # --- 分页逻辑 ---
            # 新版动态采用 offset 游标形式进行翻页
            has_more = data.get('has_more')
            next_offset = data.get('offset')

            if not has_more or not next_offset:
                print(f"UID: {host_mid} 全部动态拉取完毕。")
                break

            # 更新 offset 以供下一次循环使用
            offset = str(next_offset)

            # 适度休眠，防止请求过快被风控拦截
            time.sleep(1.5)

        except Exception as e:
            print(f"查询动态发生网络或未知错误: {e}")
            break

    # 截取恰好等于期望数量的结果返回
    return collected_dynamics[:desire_count]


def delete_user_dynamic(dyn_id_str: str, dyn_type: int, rid_str: str, cookie_str: str) -> bool:
    """
    删除指定的B站动态

    :param dyn_id_str: 动态的 ID 字符串 (从动态列表中获取)
    :param dyn_type: 动态的类型 (例如 8 表示视频动态，具体从动态列表中获取)
    :param rid_str: 资源的 ID (从动态列表中获取)
    :param cookie_str: 你的 Bilibili Cookie 字符串
    :return: 是否删除成功 (True/False)
    """

    # 1. 自动从 Cookie 中提取 csrf token (bili_jct)
    match = re.search(r'bili_jct=([^;]+)', cookie_str)
    if not match:
        print("❌ [鉴权失败] 无法从传入的 Cookie 中提取到 bili_jct (CSRF Token)，请检查 Cookie 是否完整！")
        return False
    csrf_token = match.group(1)

    # 2. 构造请求 URL 和 URL参数
    url = "https://api.bilibili.com/x/dynamic/feed/operate/remove"
    params = {
        "platform": "web",
        "csrf": csrf_token
    }

    # 3. 构造请求头
    headers = {
        "accept": "*/*",
        "accept-language": "zh-CN",
        "content-type": "application/json",
        "cookie": cookie_str,
        "origin": "https://space.bilibili.com",
        "referer": "https://space.bilibili.com/",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"
    }

    # 4. 构造 JSON 请求体 (对应抓包里的 --data-raw)
    payload = {
        "dyn_id_str": str(dyn_id_str),
        "dyn_type": int(dyn_type),
        "rid_str": str(rid_str)
    }

    # 5. 发送网络请求并捕获异常
    try:
        response = requests.post(url, params=params, headers=headers, json=payload, timeout=10)
        response.raise_for_status()  # 检查 HTTP 状态码 (200 OK)
        result = response.json()

        # 检查 B 站业务逻辑层面的状态码 (0 表示成功)
        if result.get("code") == 0:
            # print(f"✅ [删除成功]")
            return True
        else:
            print(
                f"⚠️ [删除失败] 错误码: {result.get('code')} | 接口返回: {result.get('message')}")
            return False

    except requests.exceptions.Timeout:
        print(f"⏳ [网络超时] 删除动态时请求超时。")
        return False
    except requests.exceptions.RequestException as e:
        print(f"🔌 [网络异常] 删除动态发生网络错误: {e}")
        return False
    except ValueError:
        print(f"🧩 [解析异常] 删除动态时，B 站返回的数据不是合法的 JSON 格式。")
        return False
    except Exception as e:
        print(f"💥 [未知异常] 删除动态时发生未预料的错误: {e}")
        return False


def extract_video_dynamics_info(dynamics: list) -> dict:
    """
    从动态列表中提取视频动态的关键数据。

    :param dynamics: 获取到的动态列表数据 (list of dict)
    :return: 以 bvid 为 key 的字典，包含 danmaku, play, title, pub_ts, name, mid
    """
    extracted_data = {}

    for dyn in dynamics:
        try:
            modules = dyn.get('modules', {})
            if not modules:
                continue

            # 1. 获取作者相关信息
            author_info = modules.get('module_author', {})
            pub_ts = author_info.get('pub_ts', '0')
            name = author_info.get('name', '')
            mid = author_info.get('mid', '')

            # 2. 获取动态主体内容
            module_dynamic = modules.get('module_dynamic', {})
            major = module_dynamic.get('major', {})

            # 3. 核心判断：只有类型为 MAJOR_TYPE_ARCHIVE (视频稿件) 才会有 bvid 和播放数据
            if major and major.get('type') == 'MAJOR_TYPE_ARCHIVE':
                archive = major.get('archive', {})
                bvid = archive.get('bvid')

                # 如果获取不到 bvid，直接跳过
                if not bvid:
                    continue

                title = archive.get('title', '')
                stat = archive.get('stat', {})
                danmaku = stat.get('danmaku', '0')
                play = stat.get('play', '0')

                # 4. 组装并存入结果字典
                extracted_data[bvid] = {
                    'danmaku': danmaku,
                    'play': play,
                    'title': title,
                    'pub_ts': pub_ts,
                    'name': name,
                    'mid': mid
                }

        except Exception as e:
            # 捕获单条数据解析异常，防止整个循环崩溃
            print(f"[x] 提取单条动态数据时发生异常: {e}")
            continue

    return extracted_data


def clean_old_dynamics(host_mid: int, cookie_str: str, days_old: int = 7, check_count: int = 50,
                       bvid_list: list = None) -> None:
    """
    拉取指定账号的动态，并删除超过规定天数的动态，或匹配指定 bvid 列表的动态。

    :param host_mid: 目标用户的 UID
    :param cookie_str: 你的 Bilibili Cookie 字符串
    :param days_old: 删除多少天之前的动态 (默认 7 天)
    :param check_count: 每次检查最近的多少条动态 (如果动态多，可以调大)
    :param bvid_list: 强制删除的 bvid 列表，在列表内的动态无视时间直接删除
    """
    if bvid_list is None:
        bvid_list = []

    print(f"\n🚀 开始执行动态清理任务 | UID: {host_mid} | 目标: 删除 {days_old} 天前的动态及指定bvid列表")
    exist_dynamics_info_file = r'W:\project\python_project\auto_video\config\exist_dynamics_info.json'
    exist_dynamics_info = read_json(exist_dynamics_info_file)
    # 1. 获取动态列表
    dynamics = get_user_dynamics(host_mid=host_mid, cookie_str=cookie_str, desire_count=check_count)
    dynamics_info = extract_video_dynamics_info(dynamics)
    exist_dynamics_info.update(dynamics_info)
    save_json(exist_dynamics_info_file, exist_dynamics_info)

    if not dynamics:
        print("[-] 未能获取到动态或动态列表为空，清理任务终止。")
        return
    print(f"[+] 成功获取到 {len(dynamics)} 条动态，开始检查过期及指定删除情况...")
    current_ts = int(time.time())
    threshold_sec = days_old * 24 * 3600

    # 初始化统计数据
    deleted_count = 0
    bvid_deleted_count = 0
    expired_deleted_count = 0

    # 2. 遍历动态并判断时间与bvid
    for dyn in dynamics:
        try:
            # 提取发布时间戳
            author_info = dyn.get('modules', {}).get('module_author', {})
            pub_ts_str = author_info.get('pub_ts', '0')
            pub_ts = int(pub_ts_str)

            # 计算时间差与格式化时间
            delta_sec = current_ts - pub_ts
            days_ago = delta_sec // 86400
            hours_ago = (delta_sec % 86400) // 3600
            pub_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(pub_ts))

            # 尝试提取动态标题、bvid (兼容视频、图文等不同类型)
            dyn_title = ""
            bvid = ""
            module_dynamic = dyn.get('modules', {}).get('module_dynamic', {})
            major = module_dynamic.get('major', {})

            if major and major.get('type') == 'MAJOR_TYPE_ARCHIVE':
                # 如果是视频动态，提取视频标题和bvid
                archive = major.get('archive', {})
                dyn_title = archive.get('title', '')
                bvid = archive.get('bvid', '')

            if not dyn_title:
                # 如果不是视频或者视频没标题，提取动态文本描述
                dyn_title = module_dynamic.get('desc', {}).get('text', '')

            if not dyn_title:
                dyn_title = "无标题或纯转发动态"

            # 剔除换行符以免打乱日志排版，截取前30个字符防日志过长
            dyn_title = dyn_title.replace('\n', ' ')
            if len(dyn_title) > 30:
                dyn_title = dyn_title[:30] + "..."

            # 核心判断逻辑：是否在强制删除列表内，或是否过期
            is_in_bvid_list = bool(bvid and bvid in bvid_list)
            is_expired = delta_sec > threshold_sec

            if is_in_bvid_list or is_expired:
                # 提取删除参数 (位于 modules -> module_more -> three_point_items)
                three_point_items = dyn.get('modules', {}).get('module_more', {}).get('three_point_items', [])
                del_params = None

                for item in three_point_items:
                    # 匹配删除按钮类型
                    if item.get('type') == 'THREE_POINT_DELETE':
                        del_params = item.get('params')
                        break

                if del_params:
                    target_dyn_id_str = del_params.get('dyn_id_str')
                    target_dyn_type = del_params.get('dyn_type')
                    target_rid_str = del_params.get('rid_str')

                    # 区分因为什么原因触发的删除日志
                    if is_in_bvid_list:
                        print(
                            f"[*] 发现匹配 bvid 列表的动态 (bvid: {bvid})，发布于: {pub_time_str}，标题/内容: {dyn_title}，准备删除...")
                    else:
                        print(
                            f"[*] 发现过期动态，发布于: {pub_time_str}，距今已 {days_ago} 天 {hours_ago} 小时，超过了 {days_old} 天的过期时间，标题/内容: {dyn_title}，准备删除...")

                    success = delete_user_dynamic(
                        dyn_id_str=target_dyn_id_str,
                        dyn_type=target_dyn_type,
                        rid_str=target_rid_str,
                        cookie_str=cookie_str
                    )

                    if success:
                        deleted_count += 1
                        if is_in_bvid_list:
                            bvid_deleted_count += 1
                        else:
                            expired_deleted_count += 1

                    # 强制休眠 1.5 秒，避免连续删除触发防刷机制
                    time.sleep(1.5)
                else:
                    reason_str = "在指定删除列表中" if is_in_bvid_list else "已过期"
                    print(
                        f"[-] 动态{reason_str} (发布于: {pub_time_str}，标题: {dyn_title})，但找不到删除权限 (可能是置顶/系统动态，或不是本人的)。")
            else:
                pass  # 还没到过期时间且不在列表中的动态，忽略

        except Exception as e:
            print(f"[x] 解析/处理单条动态时出错: {e}")
            continue

    print(f"🎉 任务完成 | 共检查了 {len(dynamics)} 条记录，成功删除 {deleted_count} 条动态。")
    print(f"📊 统计明细: 匹配 bvid 删除 {bvid_deleted_count} 条，正常过期删除 {expired_deleted_count} 条。\n")



# ========== 使用示例 ==========
def delete_video(bvid_list=None):
    config = init_config()
    for key, value in config.items():
        try:
            user_name = value['name']
            print(f"\n========== 开始处理用户: {user_name} (config key: {key})  bvid_list: {len(bvid_list)} ==========")
            MY_COOKIE = value['total_cookie']
            TARGET_MID = key  # 你抓包里的那个用户

            # 执行自动清理任务
            clean_old_dynamics(
                host_mid=TARGET_MID,
                cookie_str=MY_COOKIE,
                days_old=7,
                check_count=100000,
                bvid_list=bvid_list
            )
        except Exception as e:
            print(f"处理 config key: {key} 时发生错误: {e}")
            continue



if __name__ == "__main__":
    # 要处理的 config key 列表（替换为你真实的 key）
    cookie_keys = ["nana_bilibili_total_cookie", "nana_bilibili_total_cookie", "dahao_bilibili_total_cookie",
                   "dan_bilibili_total_cookie", "mama_bilibili_total_cookie", "chabian_bilibili_total_cookie"]
    cookie_keys = ["mama_bilibili_total_cookie"]




    cookie_keys = []
    config = init_config()
    for key, value in config.items():
        if 'cai' != value['name']:
            continue
        cookie_keys.append(value['total_cookie'])


    video_url = "https://www.bilibili.com/video/BV11pa5z1EEy/"
    headless = False
    watch_seconds = 6000
    max_workers = len(cookie_keys)  # 根据机器调整

    # Windows 下推荐 'spawn'
    try:
        multiprocessing.set_start_method('spawn', force=False)
    except RuntimeError:
        pass

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as exe:
        futures = {exe.submit(_worker_call, key, video_url, headless, watch_seconds): key for key in cookie_keys}
        for fut in concurrent.futures.as_completed(futures):
            key = futures[fut]
            try:
                r = fut.result()
            except Exception as e:
                r = {"key": key, "error": str(e)}
            results.append(r)

    # 打印每个任务结果
    for r in results:
        print(r)
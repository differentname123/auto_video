import json
import random
from urllib.parse import urlparse, parse_qs
from playwright.sync_api import sync_playwright

TARGET_MID = "950948"
TARGET_URL = f"https://space.bilibili.com/950948/upload/video"

# 常见真实显卡指纹池 (每次随机抽取，避免 dm_cover_img_str 撞车)
GPU_LIST = [
    "ANGLE (NVIDIA, NVIDIA GeForce RTX 3060 Direct3D11 vs_5_0 ps_5_0, or similar)",
    "ANGLE (NVIDIA, NVIDIA GeForce RTX 4090 Direct3D11 vs_5_0 ps_5_0, or similar)",
    "ANGLE (AMD, AMD Radeon RX 6700 XT Direct3D11 vs_5_0 ps_5_0, or similar)",
    "ANGLE (Intel, Intel(R) Iris(R) Xe Graphics Direct3D11 vs_5_0 ps_5_0, or similar)",
    "ANGLE (NVIDIA, NVIDIA GeForce GTX 1060 6GB Direct3D11 vs_5_0 ps_5_0, or similar)",
    "ANGLE (AMD, AMD Radeon RX 580 Series Direct3D11 vs_5_0 ps_5_0, or similar)"
]


def get_bilibili_fresh_profile():
    # print(f"[*] 准备启动全新无痕浏览器...")
    profile_data = None

    # 随机化视口分辨率，伪装不同的显示器尺寸
    viewport_width = random.choice([1920, 1600, 1536, 1440, 2560])
    viewport_height = random.choice([1080, 900, 864, 1440])
    selected_gpu = random.choice(GPU_LIST)

    with sync_playwright() as p:
        try:
            # 启动浏览器，禁用 WebRTC 防止真实 IP 泄露
            browser = p.firefox.launch(
                headless=True,
                firefox_user_prefs={
                    "media.peerconnection.enabled": False
                }
            )

            # 【核心修复 1】：强制指定国内真实语言和时区
            context = browser.new_context(
                viewport={'width': viewport_width, 'height': viewport_height},
                locale='zh-CN,zh;q=0.9,en;q=0.8',
                timezone_id='Asia/Shanghai'
            )

            page = context.new_page()
            page.set_default_timeout(60000)

            # 注入 1：隐藏 Playwright 自动化特征
            page.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

            # 【核心修复 2】：动态篡改 WebGL 指纹，彻底解决 dm_cover_img_str 相同的问题！
            webgl_spoof_js = f"""
            const getParameter = WebGLRenderingContext.prototype.getParameter;
            WebGLRenderingContext.prototype.getParameter = function(parameter) {{
                // 拦截 37446 (UNMASKED_RENDERER_WEBGL)，返回我们伪装的显卡型号
                if (parameter === 37446) return '{selected_gpu}';
                return getParameter.call(this, parameter);
            }};
            """
            page.add_init_script(webgl_spoof_js)

            # print(f"[*] 正在加载页面...")
            print(f"[*] 当前伪装分辨率: {viewport_width}x{viewport_height}")
            print(f"[*] 当前伪装显卡: {selected_gpu.split(',')[1].strip()}")

            captured_request = None

            # 监听所有请求，悄悄记录下目标接口
            def handle_request(request):
                nonlocal captured_request
                if "wbi/arc/search" in request.url and request.method == "GET":
                    captured_request = request

            page.on("request", handle_request)

            # 【核心修复 3】：让页面加载并空闲，等待指纹 JS 算完 buvid_fp
            page.goto(TARGET_URL)
            page.wait_for_load_state("networkidle", timeout=20000)

            # 额外给 3 秒钟，确保 Cookie 已经安稳落地
            page.wait_for_timeout(3000)

            # 如果首屏没触发请求，模拟人类滚轮下拉，逼迫页面加载视频列表
            if not captured_request:
                print("[*] 首次加载未捕捉到目标接口，模拟人类向下滚动...")
                page.mouse.wheel(0, 1000)
                page.wait_for_timeout(3000)

            if not captured_request:
                raise Exception("未能捕捉到 wbi/arc/search 请求，可能是页面加载失败。")

            # === 开始组装零破绽的身份档案 ===

            # 1. 组装原生 Cookie
            raw_cookies = context.cookies()
            cookie_str = "; ".join([f"{c['name']}={c['value']}" for c in raw_cookies])


            # 2. 提取并清洗 Headers
            req_headers = captured_request.all_headers()

            # 定义要排除的敏感字段（全小写匹配）
            exclude_keys = {'host', 'referer'}

            # 定义您所需要的特定大小写映射字典
            case_mapping = {
                "user-agent": "User-Agent",
                "accept": "Accept",
                "accept-language": "Accept-Language",
                "accept-encoding": "Accept-Encoding",
                "origin": "Origin",
                "sec-gpc": "Sec-GPC",
                "connection": "Connection",
                "sec-fetch-dest": "Sec-Fetch-Dest",
                "sec-fetch-mode": "Sec-Fetch-Mode",
                "sec-fetch-site": "Sec-Fetch-Site",
                "priority": "Priority",
                "cookie": "Cookie",
                "te": "TE"
            }

            # 核心修改：通过映射字典强制还原为您指定的抓包大小写格式
            headers = {}
            for k, v in req_headers.items():
                if not k.startswith(':') and k.lower() not in exclude_keys:
                    # 如果在映射表中，使用特定的格式；否则默认首字母大写
                    proper_key = case_mapping.get(k.lower(), k.title())
                    headers[proper_key] = v

            # 【关键修改】：补齐 Firefox 老身份中的灵魂指纹头，确保自洽性
            if 'TE' not in headers:
                headers['TE'] = 'trailers'
            if 'Sec-GPC' not in headers:
                headers['Sec-GPC'] = '1'
            if 'Priority' not in headers:
                headers['Priority'] = 'u=4'

            headers['Cookie'] = cookie_str

            # 3. 解析原生 URL 中的指纹参数
            parsed_url = urlparse(captured_request.url)
            query = parse_qs(parsed_url.query)

            def get_dm(key, default=""):
                return query.get(key, [default])[0]

            # 4. 组装 dm_params
            dm_params = {
                "platform": get_dm("platform", "web"),
                "web_location": get_dm("web_location", "333.1387"),
                "dm_img_list": get_dm("dm_img_list", "[]"),
                "dm_img_str": get_dm("dm_img_str", "V2ViR0wgMS"),
                "dm_cover_img_str": get_dm("dm_cover_img_str"),
                "dm_img_inter": get_dm("dm_img_inter",
                                       f'{{"ds":[],"wh":[{viewport_width},{viewport_height},15],"of":[0,0,0]}}')
            }

            # 5. 生成最终的字典
            profile_data = {
                "headers": headers,
                "dm_params": dm_params
            }

            # print("[+] 成功捕捉到完整身份特征和 Cookie！\n")

        except Exception as e:
            print(f"[!] 捕捉过程发生错误: {e}")

        finally:
            if 'browser' in locals():
                browser.close()
                # print("[*] 浏览器已销毁，不留一丝痕迹。")

    return profile_data


if __name__ == '__main__':
    new_profile = get_bilibili_fresh_profile()
    if new_profile:
        print("\n======== ✅ 完美提取结果 ========")
        print(json.dumps(new_profile, indent=4, ensure_ascii=False))
import json
from urllib.parse import urlparse, parse_qs, unquote
from typing import Optional, Dict
from playwright.sync_api import sync_playwright

TARGET_MID = "950948"
TARGET_URL = f"https://space.bilibili.com/{TARGET_MID}/upload/video"


def get_bilibili_fresh_profile() -> Optional[Dict]:
    print(f"[*] 准备启动全新无痕浏览器...")
    profile_data = None

    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(
                channel="chrome",
                # executable_path=r"C:\Program Files\Google\Chrome\Application\new_chrome.exe",
                headless=False,
                args=[
                    '--disable-blink-features=AutomationControlled',
                    '--disable-gpu',
                    '--no-sandbox',
                    '--disable-dev-shm-usage',
                ],
                ignore_default_args=["--enable-automation"]
            )

            context = browser.new_context(viewport={'width': 1920, 'height': 1080})
            page = context.new_page()
            page.set_default_timeout(60000)

            page.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

            print("[*] 正在加载页面并捕捉全新的身份指纹...")

            with page.expect_request(
                    lambda request: "wbi/arc/search" in request.url and request.method == "GET",
                    timeout=30000
            ) as req_info:
                page.goto(TARGET_URL)

            target_request = req_info.value

            # 【修复点 1】：使用 all_headers() 获取更完整的底层 Header（包含 UA 等完整特征）
            req_headers = target_request.all_headers()

            # 【修复点 2】：直接从上下文的 Cookie 罐中提取所有 Cookie，并格式化为字符串
            # 这是 100% 能拿到 buvid3, bili_ticket 等字段的最稳妥方式
            raw_cookies = context.cookies()
            cookie_str = "; ".join([f"{c['name']}={c['value']}" for c in raw_cookies])

            # 【进阶优化】：从捕获到的真实 URL 中，解析出本次绑定的 dm_ 参数
            parsed_url = urlparse(target_request.url)
            query_params = parse_qs(parsed_url.query)

            print("[+] 成功捕捉到完整身份特征和 Cookie！")

            profile_data = {
                "headers": {
                    "accept": req_headers.get("accept", "*/*"),
                    "accept-language": req_headers.get("accept-language", "zh-CN,zh;q=0.9"),
                    "origin": "https://space.bilibili.com",
                    "referer": TARGET_URL,
                    "sec-ch-ua": req_headers.get("sec-ch-ua", ""),
                    "sec-ch-ua-platform": req_headers.get("sec-ch-ua-platform", '"Windows"'),
                    "user-agent": req_headers.get("user-agent", ""),
                    "cookie": cookie_str  # <--- 这里就是完整的 Cookie 字符串了
                },
                "dm_params": {
                    "platform": "web",
                    "web_location": "333.1387",
                    # 使用 unquote 解码 URL 编码，还原真实的 JSON 字符串
                    "dm_img_list": unquote(query_params.get("dm_img_list", ["[]"])[0]),
                    "dm_img_str": unquote(query_params.get("dm_img_str", [""])[0]),
                    "dm_cover_img_str": unquote(query_params.get("dm_cover_img_str", [""])[0]),
                    "dm_img_inter": unquote(query_params.get("dm_img_inter", ["{}"])[0])
                }
            }
            print()

        except Exception as e:
            print(f"[!] 捕捉过程发生错误: {e}")

        finally:
            if 'browser' in locals():
                browser.close()
                print("[*] 浏览器已销毁，不留一丝痕迹。")

    return profile_data


if __name__ == '__main__':
    new_profile = get_bilibili_fresh_profile()
    if new_profile:
        print("\n======== ✅ 完美提取结果 ========")
        print(json.dumps(new_profile, indent=4, ensure_ascii=False))
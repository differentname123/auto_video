import asyncio
import os
import re
from gemini_webapi import GeminiClient, set_log_level
from gemini_webapi.exceptions import AuthError

set_log_level("DEBUG")

# 代理设置
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

# === 1. 将你获取到的完整 Cookie 字符串填在这里 ===
# (请务必使用最新的，且获取后已关闭浏览器窗口)
RAW_COOKIES_STR = ""
def get_cookie_value(key, text):
    """
    简单的辅助函数：从长字符串中提取指定 key 的 value
    """
    match = re.search(f"{re.escape(key)}=([^;]+)", text)
    if match:
        return match.group(1)
    return None


async def main_video_summary():
    print("正在解析 Cookie...")

    # === 2. 提取核心 Cookie ===
    # 官方文档只要求这两个
    secure_1psid = get_cookie_value("__Secure-1PSID", RAW_COOKIES_STR)
    secure_1psidts = get_cookie_value("__Secure-1PSIDTS", RAW_COOKIES_STR)

    if not secure_1psid:
        print("[错误] 无法从字符串中找到 __Secure-1PSID，请检查复制是否完整。")
        return

    print("正在初始化客户端...")

    try:
        # === 3. 修正点：在创建对象时传入 Cookie ===
        # 根据文档：client = GeminiClient(Secure_1PSID, Secure_1PSIDTS, proxy=None)
        # 注意：如果有代理，建议直接通过环境变量处理（你上面已经设了），或者在这里传 proxy 参数
        client = GeminiClient(
            secure_1psid,
            secure_1psidts,
            proxy="http://127.0.0.1:7890"
        )
        # === 4. init 只负责连接配置 ===
        # timeout=600: 只有上传大文件才需要这么久
        # auto_close=False: 保持连接
        # auto_refresh=True: 让脚本在运行时自动更新 Cookie (文档推荐)
        await client.init(timeout=600, auto_close=False, auto_refresh=True)

        print("客户端初始化成功！")

        video_path = "test1.mp4"  # 测试用

        if not os.path.exists(video_path):
            print(f"[错误] 找不到文件: {video_path}")
            return

        print(f"正在上传文件 '{video_path}' ...")

        response = await client.generate_content(
            "内容是什么。",
            model="gemini-2.5-pro",
            files=[video_path]
        )

        print("\n--- 分析结果 ---")
        print(response.text)
        print("---------------------\n")

    except AuthError:
        print("\n[致命错误] 认证失败！")
        print("1. Cookie 可能已过期（请在无痕模式重新获取并关闭窗口）。")
        print("2. 缺少 __Secure-1PSIDTS (部分账号可能没有这个值，但新版通常需要)。")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n[出错] 发生错误: {e}")


if __name__ == "__main__":
    asyncio.run(main_video_summary())
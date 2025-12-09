import asyncio
import os
from gemini_webapi import GeminiClient
from gemini_webapi.exceptions import AuthError

# 代理设置
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'


async def main_video_summary():
    client = GeminiClient()
    print("正在初始化客户端...")

    try:
        # === 修改重点在这里 ===
        # timeout=600: 将超时时间设置为 600秒 (10分钟)，防止视频上传中断
        # auto_close=False: 保持连接活跃，不自动关闭
        await client.init(timeout=600, auto_close=False)

        print("客户端初始化成功！")

        video_path = "test.mp4"  # 确保文件存在

        if not os.path.exists(video_path):
            print(f"[错误] 找不到文件: {video_path}")
            return

        print(f"正在上传视频 '{video_path}' 并等待 Gemini 处理...")
        print("提示：视频较大，请耐心等待 1-2 分钟，不要关闭程序...")

        # 再次调用生成
        response = await client.generate_content(
            "请详细总结一下这个视频的内容。如果有关键情节，请按时间顺序列出。",
            model="gemini-2.5-pro",
            files=[video_path]
        )

        print("\n--- 视频总结结果 ---")
        print(response.text)
        print("---------------------\n")

    except AuthError:
        print("\n[错误] 认证失败，请检查浏览器登录状态。")
    except Exception as e:
        # 打印完整的错误信息
        import traceback
        print(f"\n[出错] 发生错误: {e}")
        # 如果是超时，通常会显示 ReadTimeout 或 ConnectTimeout
        if "Timeout" in str(e):
            print("建议：请检查代理网络是否稳定，或者视频文件是否过大。")


if __name__ == "__main__":
    asyncio.run(main_video_summary())
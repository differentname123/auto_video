import os
import time
from typing import Tuple, Optional
from playwright.sync_api import sync_playwright, Page, expect

# ==============================================================================
# 配置区域
# ==============================================================================
# 请确保该目录没有被其他Chrome实例占用
USER_DATA_DIR = r"W:\temp\User Data"
EXECUTABLE_PATH = "C:/Program Files/Google/Chrome/Application/chrome.exe"
TARGET_URL = 'https://aistudio.google.com/'


# ==============================================================================
# 核心功能函数
# ==============================================================================

def query_google_ai_studio(prompt: str, file_path: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """
    启动浏览器，上传文件（可选），提交Prompt，并等待返回结果。

    Args:
        prompt (str): 提问的内容。
        file_path (str, optional): 附件文件的绝对路径。默认为 None。

    Returns:
        Tuple[str, str]: (error_info, response_text)
        - error_info: 如果出错，返回错误描述；否则为 None。
        - response_text: 如果成功，返回模型回答；否则为 None。
    """
    error_info = None
    response_text = None
    context = None

    print(f"--- 开始任务: Prompt='{prompt[:20]}...', File='{file_path}' ---")

    try:
        # 1. 检查文件路径（如果有）
        if file_path and not os.path.exists(file_path):
            raise FileNotFoundError(f"附件文件不存在: {file_path}")

        # 2. 启动 Playwright
        with sync_playwright() as p:
            try:
                # 启动持久化上下文
                context = p.chromium.launch_persistent_context(
                    user_data_dir=USER_DATA_DIR,
                    headless=False,  # 调试时建议开启 False，稳定后可改为 True
                    executable_path=EXECUTABLE_PATH,
                    args=['--disable-blink-features=AutomationControlled', '--start-maximized'],
                    ignore_default_args=["--enable-automation"]
                )
            except Exception as e:
                raise Exception(f"启动浏览器失败，请检查路径或确认Chrome是否已关闭: {e}")

            page = context.pages[0] if context.pages else context.new_page()
            page.set_default_timeout(60000)  # 设置默认超时时间 60秒

            # 3. 访问页面
            print("[*] 正在加载页面...")
            page.goto(TARGET_URL)

            # 4. 上传附件 (如果存在)
            if file_path:
                _upload_attachment(page, file_path)

            # 5. 提交 Prompt
            _submit_prompt(page, prompt)

            # 6. 等待并获取响应
            response_text = _wait_and_get_response(page)
            print("[+] 任务成功完成。")

    except Exception as e:
        error_info = str(e)
        print(f"[!] 执行过程中发生错误: {error_info}")
        # 可选：出错时截图
        # if context and context.pages:
        #     context.pages[0].screenshot(path="error_last_state.png")

    finally:
        # 7. 清理资源
        if context:
            try:
                context.close()
                print("[*] 浏览器环境已关闭。")
            except Exception:
                pass

    return error_info, response_text


# ==============================================================================
# 内部辅助函数 (私有)
# ==============================================================================

def _upload_attachment(page: Page, file_path: str):
    """(内部调用) 上传附件逻辑"""
    print(f"[*] 正在上传附件: {os.path.basename(file_path)}")

    # 点击加号
    with page.expect_file_chooser(timeout=15000) as fc_info:
        attachment_button = page.locator('[aria-label="Insert images, videos, audio, or files"]')
        attachment_button.click()

        # 点击 "Upload a file"
        upload_option = page.get_by_role("menuitem", name="Upload a file")
        upload_option.click()

    file_chooser = fc_info.value
    file_chooser.set_files(file_path)

    # 等待上传转圈消失 (Important)
    spinner = page.locator(".upload-spinner")
    expect(spinner).to_be_hidden(timeout=60000)
    print("[+] 附件上传完毕。")


def _submit_prompt(page: Page, prompt: str):
    """(内部调用) 填写并提交Prompt"""
    print("[*] 正在提交Prompt...")

    # 定位输入框
    prompt_input = page.get_by_placeholder("Start typing a prompt")
    expect(prompt_input).to_be_editable(timeout=15000)
    prompt_input.fill(prompt)

    # 定位并点击 Run 按钮
    run_button = page.get_by_role("button", name="Run", exact=True)
    expect(run_button).to_be_enabled(timeout=15000)
    run_button.click()


def _wait_and_get_response(page: Page) -> str:
    """(内部调用) 等待流式输出结束并提取文本"""
    print("[*] 等待模型响应中...")

    # 1. 确认开始生成：等待 "Stop" 按钮出现
    # 策略：找到文本为 Stop 的按钮
    stop_btn = page.locator("button").filter(has_text="Stop")
    expect(stop_btn).to_be_visible(timeout=30000)

    # 2. 确认生成结束：等待 "Stop" 按钮消失 (最长等待5分钟)
    expect(stop_btn).to_be_hidden(timeout=300000)

    # 3. 提取内容
    # 获取最后一个 Model 的回复框
    response_container = page.locator('[data-turn-role="Model"]').last
    expect(response_container).to_be_visible()

    return response_container.inner_text()


# ==============================================================================
# 使用示例
# ==============================================================================

if __name__ == '__main__':
    # 测试文件路径
    test_file = r"W:\project\python_project\watermark_remove\common_utils\video_scene\test.jpg"
    test_prompt = "请详细描述这张图片的内容。"

    # 调用封装好的函数
    err, response = query_google_ai_studio(prompt=test_prompt, file_path=test_file)

    if err:
        print("\n======== ❌ 失败 ========")
        print(f"错误信息: {err}")
    else:
        print("\n======== ✅ 成功 ========")
        print("模型回复内容:")
        print(response)
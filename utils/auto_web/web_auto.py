import json
import os
import re
import shutil
import time
import argparse
import sys
from typing import Tuple, Optional
from playwright.sync_api import sync_playwright, Page, expect
from playwright.sync_api import Page, expect, Locator

import time
import datetime
import sys
import csv
import os
import traceback  # 用于捕获更详细的异常信息

# ==============================================================================
# 配置区域
# ==============================================================================
# 用于保存浏览器登录状态的目录，请确保该目录可写
# 第一次运行登录后，这里会生成包含cookies等信息的文件
USER_DATA_DIR = r"W:\temp\new_taobao13"
TARGET_URL_BASE = 'https://aistudio.google.com/prompts/new_chat'


# ==============================================================================
# 核心功能函数
# ==============================================================================

def _get_dir_size(start_path='.'):
    """计算目录总大小 (返回字节数)"""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(start_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                # 跳过链接文件，避免重复计算
                if not os.path.islink(fp):
                    try:
                        total_size += os.path.getsize(fp)
                    except Exception:
                        pass
    except Exception:
        pass
    return total_size


def _format_size(size):
    """将字节转换为易读的格式 (MB, GB)"""
    power = 1024
    n = size
    power_labels = {0: '', 1: 'KB', 2: 'MB', 3: 'GB', 4: 'TB'}
    count = 0
    while n > power:
        n /= power
        count += 1
    return f"{n:.2f} {power_labels.get(count, 'B')}"


def clean_browser_cache(user_data_dir: str):
    """
    深度清理 Chromium 用户目录缓存，并显示清理前后的体积变化。
    保留 Cookies, LocalStorage 以维持登录状态。
    """
    if not os.path.exists(user_data_dir):
        print(f"[-] 目录不存在，无需清理: {user_data_dir}")
        return

    print("=" * 40)
    print("🚀 正在执行浏览器数据瘦身...")

    # 1. 计算清理前大小
    size_before = _get_dir_size(user_data_dir)
    print(f"[*] 清理前占用空间: {_format_size(size_before)}")

    # 2. 定义垃圾目录清单 (这些目录删除后不会影响登录状态)
    # 这些目录可能直接在 user_data_dir 下，也可能在 Default 子目录下
    garbage_targets = [
        "Cache",  # 网页缓存 (图片/CSS/JS)
        "Code Cache",  # 编译后的JS代码缓存
        "GPUCache",  # GPU渲染缓存
        "ShaderCache",  # 着色器缓存
        "GrShaderCache",  # 图形资源缓存
        "Service Worker",  # 服务工作线程 (Google系网页这块特别大)
        "CacheStorage",  # 离线缓存
        "ScriptCache",  # 脚本缓存
        "Crashpad",  # 崩溃转储日志 (Dumps)
        "BrowserMetrics",  # 浏览器指标数据
        "Safe Browsing",  # 安全浏览数据库
        "blob_storage",  # Blob数据
        "OptimizationGuidePredictionModels",  # 预测模型缓存
    ]

    # 扫描根目录和 Default 子目录
    scan_paths = [user_data_dir, os.path.join(user_data_dir, "Default")]

    deleted_count = 0

    for base_path in scan_paths:
        if not os.path.exists(base_path):
            continue

        for target in garbage_targets:
            target_full_path = os.path.join(base_path, target)

            if os.path.exists(target_full_path):
                try:
                    # 如果是文件夹则递归删除，如果是文件则直接删除
                    if os.path.isdir(target_full_path):
                        shutil.rmtree(target_full_path, ignore_errors=True)
                    else:
                        os.remove(target_full_path)
                    deleted_count += 1
                except Exception as e:
                    # 遇到文件占用(PermissionError)直接跳过，不打印骚扰信息
                    pass

    # 3. 计算清理后大小
    size_after = _get_dir_size(user_data_dir)
    freed_size = size_before - size_after

    print(f"[*] 清理后占用空间: {_format_size(size_after)}")
    print(f"[+] 成功释放空间:   {_format_size(freed_size)} (清理了 {deleted_count} 个项目)")
    print("=" * 40)


class PageCrashedException(Exception):
    """自定义异常，用于表示页面已崩溃。"""
    pass


def check_for_crash_and_abort(page: Page):
    """
    (内部调用) 快速检查页面是否崩溃。如果崩溃，则立即抛出异常以终止任务。
    """
    try:
        # 查找崩溃页面的特征元素：“重新加载”按钮。
        # 在简体中文环境下，按钮文本是 "重新加载"。
        reload_button = page.get_by_role("button", name="重新加载")

        # 使用极短的超时来检查，因为它应该立即存在于崩溃页面上。
        # 如果页面正常，这个检查会很快失败，不会浪费时间。
        if reload_button.is_visible(timeout=1000):  # 1秒超时
            error_msg = "页面已崩溃 (检测到 '重新加载' 按钮)，任务终止。"
            print(f"[!] {error_msg}")
            # 抛出自定义异常，这样我们可以在主逻辑中捕获它并进行处理。
            raise PageCrashedException(error_msg)

    except Exception as e:
        traceback.print_exc()
        # 如果在1秒内找不到按钮 (抛出 TimeoutError)，或者发生其他错误，
        # 都意味着页面大概率是正常的，我们可以安全地忽略这个异常。
        # 我们只关心 PageCrashedException。
        if isinstance(e, PageCrashedException):
            raise  # 将我们自己的异常重新抛出
        # 其他异常（如 TimeoutError）则忽略
        pass


def login_and_save_session(model_name: str = "gemini-2.5-pro"):
    """
    启动浏览器，让用户手动登录，并将登录会话保存到 USER_DATA_DIR。
    """
    print("--- 启动浏览器进行手动登录 ---")
    print(f"会话信息将保存在: {USER_DATA_DIR}")
    clean_browser_cache(USER_DATA_DIR)
    with sync_playwright() as p:
        # 使用自带的 chromium，并启动持久化上下文
        context = p.chromium.launch_persistent_context(
            user_data_dir=USER_DATA_DIR,
            headless=False,  # 必须为 False 以便用户可以看到和操作浏览器
            args=['--disable-blink-features=AutomationControlled', '--start-maximized', '--disable-gpu',
                  '--disk-cache-size=1',
                  '--window-position=0,0',
                  '--media-cache-size=1',
                  '--disable-application-cache',
                  '--disable-component-update', ],
            ignore_default_args=["--enable-automation"]
        )

        page = context.new_page()
        target_url = f"{TARGET_URL_BASE}?model={model_name}"
        page.goto(target_url)

        print("\n" + "=" * 60)
        print("浏览器已打开。请在浏览器窗口中手动完成登录操作。")
        print("登录成功并进入AI Studio主界面后，请回到本命令行窗口，然后按 Enter 键继续...")
        print("=" * 60)

        # 阻塞程序，等待用户在命令行按 Enter
        input()

        # 用户按 Enter 后，关闭浏览器，此时登录状态已自动保存到 USER_DATA_DIR
        context.close()
        print("\n[+] 登录会话信息已成功保存。现在可以使用 'query' 命令来运行任务了。")


def click_acknowledge_if_present(page: Page):
    """
    检查并点击 "Acknowledge" 弹窗按钮。
    """
    # 优先尝试匹配 aria-label，这是最准确的 Role 匹配方式
    # 或者使用 page.locator("button", has_text="Acknowledge")
    acknowledge_button = page.get_by_role("button", name="Agree to the copyright acknowledgement")
    time.sleep(2)
    try:
        # is_visible 会自动轮询等待元素出现，直到超时。
        # 这里不需要 time.sleep，直接依靠 timeout。
        if acknowledge_button.is_visible(timeout=5000):
            print("[+] 检测到 'Acknowledge' 按钮，正在点击...")
            acknowledge_button.click()

            # 确认点击成功（按钮消失）
            expect(acknowledge_button).to_be_hidden(timeout=5000)
            print("[+] 'Acknowledge' 弹窗已处理。")
        else:
            print("[-] 未发现 'Acknowledge' 弹窗，继续执行。")

    except Exception as e:
        # 捕获可能的报错，比如元素在判断可见后瞬间消失导致 click 失败
        print(f"[-] 处理 'Acknowledge' 弹窗时发生异常: {e}")

def query_google_ai_studio(prompt: str, file_path: Optional[str] = None, user_data_dir=USER_DATA_DIR,
                           model_name: str = "gemini-2.5-pro") -> Tuple[Optional[str], Optional[str]]:
    """
    使用已保存的登录会话启动浏览器，上传文件（可选），提交Prompt，并等待返回结果。

    Args:
        prompt (str): 提问的内容。
        user_data_dir (str): 保存浏览器登录状态的用户数据目录。
        file_path (str, optional): 附件文件的绝对路径。默认为 None。

    Returns:
        Tuple[str, str]: (error_info, response_text)
        - error_info: 如果出错，返回错误描述；否则为 None。
        - response_text: 如果成功，返回模型回答；否则为 None。
    """
    # 检查登录会话是否存在
    if not os.path.isdir(user_data_dir):  # <-- 使用传入的参数
        error_msg = f"用户数据目录不存在: {user_data_dir}\n请先运行 'python {os.path.basename(__file__)} login --user-data-dir <你的目录>' 命令进行登录。"
        return error_msg, None

    error_info = None
    response_text = None
    context = None

    print(
        f"--- 开始任务: Prompt='{prompt[:20]}...', File='{file_path}' --- 当前时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # 1. 检查文件路径（如果有）
        if file_path and not os.path.exists(file_path):
            raise FileNotFoundError(f"附件文件不存在: {file_path}")

        # 2. 启动 Playwright
        with sync_playwright() as p:
            try:
                # 启动持久化上下文，它会自动加载 user_data_dir 中的登录信息
                #
                context = p.chromium.launch_persistent_context(
                    user_data_dir=user_data_dir,
                    headless=False,  # 必须保持 False 以通过反爬检测

                    # 显式指定视口大小，替代 start-maximized
                    # 因为移到屏幕外后，最大化可能失效或导致渲染布局异常
                    viewport={'width': 1920, 'height': 1080},

                    args=[
                        '--disable-blink-features=AutomationControlled',
                        '--disable-gpu',

                        # 【核心修改点】: 将窗口位置移动到屏幕显示范围之外
                        '--window-position=-10000,-10000',

                        # 移除 '--start-maximized'，因为我们要手动控制视口大小且移出屏幕
                        # '--start-maximized',

                        '--no-sandbox',
                        '--disable-dev-shm-usage'
                    ],
                    ignore_default_args=["--enable-automation"]
                )

                # # 下面这个是能够看到窗口的模式
                # context = p.chromium.launch_persistent_context(
                #     user_data_dir=user_data_dir, # <-- 使用传入的参数
                #     headless=False,  # 调试时建议开启 False，稳定后可改为 True
                #     args=['--disable-blink-features=AutomationControlled', '--start-maximized', '--disable-gpu',    '--window-position=0,0'],
                #     ignore_default_args=["--enable-automation"]
                # )

            except Exception as e:
                raise Exception(f"启动浏览器失败，请检查或确认浏览器是否已关闭: {e}")

            page = context.pages[0] if context.pages else context.new_page()
            page.set_default_timeout(60000)  # 设置默认超时时间 60秒

            # 3. 访问页面
            print("[*] 正在加载页面...")
            target_url = f"{TARGET_URL_BASE}?model={model_name}"
            # target_url = f"{TARGET_URL_BASE}?model=gemini-3-pro-preview"
            # target_url = f"{TARGET_URL_BASE}?model=gemini-2.5-pro"


            page.goto(target_url)
            # time.sleep(1000)
            # [修改] 页面加载后立即检查崩溃
            check_for_crash_and_abort(page)

            # 4. 处理弹窗和上传附件
            click_acknowledge_if_present(page)

            if file_path:
                # [修改] 操作前再次检查
                check_for_crash_and_abort(page)
                _upload_attachment(page, file_path)

            # [修改] 提交前也检查一下，确保页面状态良好
            check_for_crash_and_abort(page)

            for i in range(3):
                # 5. 提交 Prompt
                click_acknowledge_if_present(page)
                _submit_prompt(page, prompt)
                # 6. 等待并获取响应
                response_text = _wait_and_get_response(page)
                if "An internal error has occurred." not in response_text:
                    break
                time.sleep(2)
                print("[-] 检测到内部错误，正在重试...")
            print(f"[+] 任务成功任务完成任务。{file_path} {response_text[:100]}...")

    # [修改] 新增对页面崩溃异常的捕获
    except PageCrashedException as crash_e:
        error_info = str(crash_e)
        # 崩溃时截图
        if context and context.pages:
            try:
                screenshot_path = f"crash_screenshot_{int(time.time())}.png"
                # 确保有页面可以截图
                if context.pages:
                    context.pages[0].screenshot(path=screenshot_path)
                    print(f"[*] 崩溃截图已保存至: {screenshot_path}")
            except Exception as screenshot_e:
                print(f"[!] 截取崩溃快照失败: {screenshot_e}")

    except Exception as e:
        error_info = str(e)
        print(f"[!] 执行过程中发生错误: {error_info[:1000]} {file_path}")
        # 可选：出错时截图
        if context and context.pages:
            try:
                screenshot_path = f"error_screenshot_{int(time.time())}.png"
                if context.pages:
                    context.pages[0].screenshot(path=screenshot_path)
                    print(f"[*] 错误截图已保存至: {screenshot_path}")
            except Exception as screenshot_e:
                print(f"[!] 截取错误快照失败: {screenshot_e}")

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
# 内部辅助函数 (与原版相同，无需修改)
# ==============================================================================

def _upload_attachment(page: Page, file_path: str):
    """(内部调用) 上传附件逻辑 (两步点击均已强化，具备高兼容性)"""
    print(f"[*] 正在上传附件: {os.path.basename(file_path)}")
    click_acknowledge_if_present(page)

    with page.expect_file_chooser(timeout=15000) as fc_info:
        # --- 第 1 步: 点击主附件按钮 (已强化) ---
        best_locator = page.locator('[data-test-add-chunk-menu-button]')
        fallback_locator = page.get_by_role(
            "button",
            name=re.compile(r"(?=.*images)(?=.*videos)(?=.*audio)(?=.*files)", re.IGNORECASE)
        )
        attachment_button = best_locator.or_(fallback_locator)
        attachment_button.click()

        # --- 第 2 步: 点击"上传文件"菜单项 (新增的强化) ---
        # 使用正则表达式来兼容 "Upload a file" 和 "Upload File" 等多种写法
        upload_option = page.get_by_role(
            "menuitem",
            name=re.compile(r"Upload (a )?file", re.IGNORECASE)
        )
        upload_option.click()

    file_chooser = fc_info.value
    file_chooser.set_files(file_path)

    spinner = page.locator(".upload-spinner")
    expect(spinner).to_be_hidden(timeout=60000)
    print("[+] 附件上传完毕。")


def _remove_google_grounding(page: Page):
    """
    (内部调用) 检查是否存在 'Remove Grounding with Google Search' 按钮，
    如果有则点击关闭。
    """
    try:
        # 根据 HTML 中的 aria-label="Remove Grounding with Google Search" 定位按钮
        grounding_close_btn = page.get_by_role("button", name="Remove Grounding with Google Search")

        # 使用 short timeout (例如 1-2秒) 快速检查可见性
        # 我们不希望因为按钮不存在而卡住脚本太久
        if grounding_close_btn.is_visible(timeout=2000):
            print("[*] 检测到 Google Grounding 关联，正在移除...")
            grounding_close_btn.click()

            # 可选：稍微等待一下确认点击生效，防止 UI 动画干扰
            page.wait_for_timeout(500)
    except Exception as e:
        # 这是一个非阻塞操作，如果出错（比如元素刚好消失），打印日志但不中断流程
        print(f"[!] 检查 Google Grounding 按钮时出现轻微异常 (已忽略): {e}")

def _submit_prompt(page: Page, prompt: str):
    """(内部调用) 填写并提交Prompt (已升级为高兼容性定位器)"""
    print("[*] 正在提交Prompt...")

    # --- 1. 定位输入框 (替换旧的 get_by_placeholder) ---
    prompt_input: Locator | None = None
    try:
        # 步骤 A: 基础筛选 - 获取所有可见的 textbox 元素
        all_textboxes = page.get_by_role("textbox").filter(has_not_text="hidden").all()

        if not all_textboxes:
            raise Exception("在页面上找不到任何可见的输入框 (role='textbox')。")

        # 步骤 B: 位置筛选 - 过滤出位于页面下半部分的
        viewport_height = page.viewport_size['height']
        lower_half_textboxes = [
            box for box in all_textboxes
            if box.bounding_box()['y'] > viewport_height / 3
        ]

        if not lower_half_textboxes:
            raise Exception("在页面的下半部分找不到任何可见的输入框。")

        # 步骤 C: 智能决胜
        if len(lower_half_textboxes) == 1:
            prompt_input = lower_half_textboxes[0]
        else:
            # 优先选择 aria-label 包含关键意图词的
            tie_breaker_keywords = re.compile("prompt|type|enter|start typing", re.IGNORECASE)

            preferred_candidates = [
                box for box in lower_half_textboxes
                if tie_breaker_keywords.search(box.get_attribute("aria-label") or "")
            ]

            if len(preferred_candidates) == 1:
                prompt_input = preferred_candidates[0]
            else:
                # 备用策略：选择最后一个
                prompt_input = lower_half_textboxes[-1]

    except Exception as e:
        print(f"[!] 错误：定位Prompt输入框失败。 {e}")
        # 如果你想兼容旧版，可以在这里回退到旧的定位器
        print("[!] 尝试使用旧的 placeholder 定位器作为备用方案...")
        prompt_input = page.get_by_placeholder("Start typing a prompt")

    # --- 2. 填写Prompt (与原代码一致) ---
    expect(prompt_input).to_be_editable(timeout=15000)
    prompt_input.fill(prompt)

    # --- 3. 定位并点击运行按钮 (增加对 "Generate" 的兼容) ---
    # 使用正则表达式同时匹配 "Run" 和 "Generate"
    run_button = page.get_by_role(
        "button",
        name=re.compile(r"^(Run|Generate)$", re.IGNORECASE)
    )

    expect(run_button).to_be_enabled(timeout=300000)
    _remove_google_grounding(page)
    run_button.click()


def _scroll_page_to_bottom(page: Page, steps: int = 20, step_px: int = 1500, delay: float = 0.05):
    """不看任何容器，直接强制往下滚页面，保证滚到最底部"""
    for _ in range(steps):
        try:
            # 将鼠标移到视口中央
            vp = page.viewport_size or {"width": 1280, "height": 720}
            page.mouse.move(vp["width"] / 2, vp["height"] / 2)
            # 滑轮往下滑
            page.mouse.wheel(0, step_px)
        except:
            pass
        time.sleep(delay)
    # 最后用一次键盘 END 强制到底
    try:
        page.keyboard.press("End")
    except:
        pass


def _wait_and_get_response(page: Page) -> str:
    """(内部调用) 等待流式输出结束并提取文本"""
    print("[*] 等待模型响应中...")
    stop_btn = page.locator("button").filter(has_text="Stop")
    expect(stop_btn).to_be_visible(timeout=30000)
    expect(stop_btn).to_be_hidden(timeout=300000)
    _scroll_page_to_bottom(page, steps=40)  # 再滚到底，确保看到最后生成的节点
    time.sleep(1)  # 等待1秒，确保内容稳定
    response_container = page.locator('[data-turn-role="Model"]').last

    expect(response_container).to_be_visible()
    return response_container.inner_text()


# ==============================================================================
# ==============================================================================
# 程序主入口和使用示例
# ==============================================================================
if __name__ == '__main__':
    login_and_save_session()

    # 测试文件路径
    test_file = r"W:\project\python_project\watermark_remove\common_utils\video_scene\test.jpg"
    test_prompt = "请详细描述这张图片的内容。"

    # 调用封装好的函数12
    err, response = query_google_ai_studio(prompt=test_prompt, file_path=test_file)

    if err:
        print("\n======== ❌ 失败 ========")
        print(f"错误信息: {err}")
    else:
        print("\n======== ✅ 成功 ========")
        print("模型回复内容:")
        print(response)

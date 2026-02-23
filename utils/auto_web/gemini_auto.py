# -- coding: utf-8 --
""":authors:
    zhuxiaohu
:create_date:
    2025/12/12 20:23
:last_date:
    2025/12/12 20:23
:description:
    Gemini 自动化调用管理器（Playwright版）
"""
import json
import os
import threading
import time
import random
import traceback
from datetime import datetime
from pathlib import Path

from utils.auto_web.web_auto import query_google_ai_studio
from utils.common_utils import read_json, save_json

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_FILE = BASE_DIR / 'config/gemini_auto.json'
STATS_FILE = BASE_DIR / 'config/gemini_auto_stats.json'

# 全局默认最大并发数 (如果配置文件中没有设置 max_concurrency，则使用此值)
# 根据你的电脑性能调整，建议：8G内存设为 2-3，16G内存设为 4-6
DEFAULT_MAX_CONCURRENCY = 2

# 【新增配置】每个账号连续使用的默认次数
DEFAULT_USAGE_STREAK_LIMIT = 10


# ==========================================
# 2. 基础工具类：文件锁 (保持不变)
# ==========================================

class SimpleFileLock:
    def __init__(self, lock_file, timeout=10):
        self.lock_file = lock_file
        self.timeout = timeout
        self.fd = None

    def __enter__(self):
        start_time = time.time()
        while True:
            try:
                # 尝试以独占模式创建文件作为锁
                self.fd = os.open(self.lock_file, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                return self
            except FileExistsError:
                if time.time() - start_time > self.timeout:
                    raise TimeoutError(f"获取文件锁超时: {self.lock_file}")
                time.sleep(0.1)

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if self.fd:
                os.close(self.fd)
            if os.path.exists(self.lock_file):
                os.remove(self.lock_file)
        except OSError:
            pass


# ==========================================
# 3. 核心：Playwright 账号管理器
# ==========================================

# ==========================================
# 3. 核心：Playwright 账号管理器
# ==========================================

class PlaywrightAccountManager:
    def __init__(self, config_path, stats_path):
        self.config_path = config_path
        self.stats_path = stats_path
        self.lock_path = str(stats_path) + ".lock"

    def _check_and_reset_stuck_accounts(self, stats_data, timeout_seconds=900):
        """检查并重置长时间处于 using 状态的账号"""
        now = datetime.now()
        for name, info in stats_data.items():
            if not isinstance(info, dict) or 'status' not in info:
                continue
            if info.get('status') == 'using':
                # 兼容新旧数据结构
                last_time_str = info.get('account_last_used_time', info.get('last_used_time', ''))
                if last_time_str:
                    try:
                        last_time = datetime.strptime(last_time_str, "%Y-%m-%d %H:%M:%S")
                        if (now - last_time).total_seconds() > timeout_seconds:
                            print(f"[Manager] 账号 {name} 'using'状态超时({timeout_seconds}s)，强制重置为'idle'")
                            info['status'] = 'idle'
                            cur_model = info.get('current_using_model')

                            # 将错误信息精确记录到对应的模型下
                            if cur_model and 'models' in info and cur_model in info['models']:
                                info['models'][cur_model]['last_error_info'] = "System: Force reset due to timeout"
                            else:
                                info['last_error_info'] = "System: Force reset due to timeout"
                    except ValueError:
                        info['status'] = 'idle'
                        info['last_error_info'] = "System: Force reset due to invalid time format"
        return stats_data

    def allocate_account(self, model_name=None):
        """
        分配一个空闲账号。
        引入多模型维度管理，同一账号的不同 model_name 限流与冷却互不影响。
        """
        _m_name = model_name or "default_model"

        with SimpleFileLock(self.lock_path):
            # 1. 读取配置和统计信息
            raw_config = read_json(self.config_path)
            stats = read_json(self.stats_path)
            config_list = raw_config.get('account_list', [])

            # 2. 获取配置参数
            max_concurrency = raw_config.get('max_concurrency', DEFAULT_MAX_CONCURRENCY)
            usage_streak_limit = raw_config.get('usage_streak_limit', DEFAULT_USAGE_STREAK_LIMIT)

            # 3. 数据预处理
            valid_accounts_map = {
                item['name']: item.get('user_data_dir', '')
                for item in config_list if item.get('name') and item.get('user_data_dir')
            }

            # 3.1 基础同步与清理 (兼容多模型 pool 键)
            for name in list(stats.keys()):
                if name.startswith('active_pool'): continue
                if name not in valid_accounts_map: del stats[name]

            # 初始化数据结构，引入 models 子字典
            for name in valid_accounts_map:
                if name not in stats:
                    stats[name] = {"status": "idle", "account_last_used_time": "", "current_using_model": None,
                                   "models": {}}
                if "models" not in stats[name]:
                    stats[name]["models"] = {}
                if _m_name not in stats[name]["models"]:
                    stats[name]["models"][_m_name] = {
                        "last_used_time": "", "last_error_info": None, "total_usage": 0, "current_streak": 0
                    }

            # 3.2 超时重置
            stats = self._check_and_reset_stuck_accounts(stats)

            # 4. 前置全局并发检查 (全局使用中的总数)
            current_using_total = sum(
                1 for name, info in stats.items() if isinstance(info, dict) and info.get('status') == 'using')
            if current_using_total >= max_concurrency:
                save_json(self.stats_path, stats)
                return None, None

            # 5. 维护该模型专属的活跃池 (Active Pool)
            pool_key = f"active_pool_{_m_name}"
            exhausted_accounts = set()
            for name, info in stats.items():
                if isinstance(info, dict) and 'models' in info:
                    m_info = info['models'].get(_m_name, {})
                    if m_info.get('current_streak', 0) >= usage_streak_limit:
                        m_info['current_streak'] = 0  # 达到上限，准备轮换
                        exhausted_accounts.add(name)

            active_pool = stats.get(pool_key, [])
            # 过滤掉已失效或该模型下已用完次数的账号
            active_pool = [n for n in active_pool if n in valid_accounts_map and n not in exhausted_accounts]

            # 如果池不满，补充新账号
            needed = max_concurrency - len(active_pool)
            if needed > 0:
                # 定义冷却时间 30分钟
                COOLDOWN_SECONDS = 1200
                now_dt = datetime.now()

                def is_cooldown_ready(acc_name):
                    """检查该账号在该模型下的上次使用时间是否在30分钟前"""
                    m_info = stats.get(acc_name, {}).get('models', {}).get(_m_name, {})
                    last_time_str = m_info.get('last_used_time', '')
                    if not last_time_str:
                        return True
                    try:
                        last_time = datetime.strptime(last_time_str, "%Y-%m-%d %H:%M:%S")
                        return (now_dt - last_time).total_seconds() > COOLDOWN_SECONDS
                    except ValueError:
                        return True

                candidates = [
                    n for n in valid_accounts_map
                    if n not in active_pool
                       and is_cooldown_ready(n)
                ]

                # 按该模型的总使用次数排序，优先使用少的
                candidates.sort(key=lambda n: stats.get(n, {}).get('models', {}).get(_m_name, {}).get('total_usage', 0))
                active_pool.extend(candidates[:needed])

            stats[pool_key] = active_pool

            # 6. 从活跃池中选择一个空闲账号分配（注意：status=='idle' 是账号级限制，防止多开同一浏览器）
            target_name = None
            idle_in_pool = [n for n in active_pool if stats.get(n, {}).get('status') == 'idle']

            if idle_in_pool:
                idle_in_pool.sort(
                    key=lambda n: stats.get(n, {}).get('models', {}).get(_m_name, {}).get('total_usage', 0))
                target_name = idle_in_pool[0]

            if not target_name:
                save_json(self.stats_path, stats)
                return None, None

            # 7. 更新状态
            target_info = stats[target_name]
            target_info['status'] = 'using'
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            target_info['account_last_used_time'] = now_str
            target_info['current_using_model'] = _m_name

            # 模型级状态更新
            m_info = target_info['models'][_m_name]
            m_info['last_used_time'] = now_str
            m_info['total_usage'] = m_info.get('total_usage', 0) + 1
            m_info['current_streak'] = m_info.get('current_streak', 0) + 1

            save_json(self.stats_path, stats)
            return target_name, valid_accounts_map[target_name]

    def release_account(self, account_name, error_info=None):
        """释放账号，重置为 idle，错误信息精确定位到模型。"""
        with SimpleFileLock(self.lock_path):
            stats = read_json(self.stats_path)

            if account_name in stats and isinstance(stats[account_name], dict):
                info = stats[account_name]
                info['status'] = 'idle'
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                cur_model = info.get('current_using_model')
                if cur_model and 'models' in info and cur_model in info['models']:
                    m_info = info['models'][cur_model]
                    m_info['last_used_time'] = now_str
                    m_info['last_error_info'] = str(error_info)[:1000] if error_info else None
                    if error_info:
                        print(
                            f"[Manager] 账号 {account_name} 在模型 {cur_model} 出现错误，重置其连续使用次数。{error_info}")
                        m_info['current_streak'] = 0
                else:
                    # 兼容后备逻辑
                    info['last_error_info'] = str(error_info)[:1000] if error_info else None
                    if error_info:
                        print(f"[Manager] 账号 {account_name} 出现错误，重置其连续使用次数。{error_info}")
                        info['current_streak'] = 0

            save_json(self.stats_path, stats)


# ==========================================
# 4. 对外统一接口 (保持不变)
# ==========================================

# 确保配置目录存在
if not CONFIG_FILE.parent.exists():
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)

manager = PlaywrightAccountManager(str(CONFIG_FILE), str(STATS_FILE))


def generate_gemini_content_playwright(prompt, file_path=None, wait_timeout=600, model_name="gemini-2.5-pro"):
    """
    使用 Playwright 账号管理器安全地调用 Gemini。
    """
    pid = os.getpid()
    tid = threading.get_ident()
    log_prefix = f"[System][PID:{pid},TID:{tid}] {file_path}"

    start_time = time.time()
    account_name, user_data_dir = None, None

    no_file_name_list = ['new_taobao6', 'new_taobao9']
    # 1. 循环申请账号
    while time.time() - start_time < wait_timeout:
        account_name, user_data_dir = manager.allocate_account(model_name=model_name)
        if file_path and user_data_dir and any(x in user_data_dir for x in no_file_name_list):
            account_name, user_data_dir = None, None
        if account_name:
            break

        elapsed = int(time.time() - start_time)
        # 这里把等待日志的频率稍微降低一点，或者简单打印
        if elapsed % 10 == 0:
            print(f"{log_prefix} 资源繁忙(并发已满或无空闲号)，等待中... (已等待 {elapsed}s / {wait_timeout}s)")

        # 随机等待一段时间再重试
        time.sleep(random.uniform(5, 15))

    if not account_name:
        return f"System Busy: 等待 {wait_timeout} 秒后仍无可用资源。", None

    print(f"{log_prefix} 分配账号: {account_name} ({os.path.basename(user_data_dir)}) {model_name} {file_path}")

    error_detail, result_text = None, None
    try:
        file_to_upload = None
        if file_path:
            if isinstance(file_path, list) and file_path:
                file_to_upload = file_path[0]
            elif isinstance(file_path, str):
                file_to_upload = file_path

        # 2. 调用核心函数
        error_detail, result_text = query_google_ai_studio(
            prompt=prompt,
            file_path=file_to_upload,
            user_data_dir=user_data_dir,
            model_name=model_name
        )
    except Exception as e:
        error_detail = f"管理器外部发生严重错误: {str(e)}\n\n{traceback.format_exc()}"
    finally:
        # 3. 释放账号
        print(f"{log_prefix} 释放账号: {account_name}")

        # 如果返回包含 rate limit 的提示，精准惩罚对应的模型配额
        try:
            if result_text and "You've reached your rate limit. Please try again later" in result_text:
                with SimpleFileLock(manager.lock_path):
                    stats = read_json(manager.stats_path)
                    raw_config = read_json(manager.config_path) or {}
                    usage_streak_limit = raw_config.get('usage_streak_limit', DEFAULT_USAGE_STREAK_LIMIT)

                    if account_name in stats and isinstance(stats[account_name], dict):
                        info = stats[account_name]
                        _m_name = model_name or "default_model"

                        # 确保基础结构存在
                        if 'models' not in info:
                            info['models'] = {}
                        if _m_name not in info['models']:
                            info['models'][_m_name] = {}

                        m_info = info['models'][_m_name]
                        cur = m_info.get('current_streak', 0)

                        # 将该模型的 streak 置为上限，惩罚触发轮换和冷却
                        if cur < usage_streak_limit:
                            inc = usage_streak_limit - cur
                            m_info['current_streak'] = usage_streak_limit
                            m_info['total_usage'] = m_info.get('total_usage', 0) + inc

                        # 解除全局浏览器锁定，但更新模型错误日志
                        info['status'] = 'idle'
                        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        info['account_last_used_time'] = now_str
                        m_info['last_used_time'] = now_str
                        m_info['last_error_info'] = "Remote: rate limit detected"

                    save_json(manager.stats_path, stats)
                print(
                    f"{log_prefix} 账号 {account_name} 在模型 {_m_name} 检测到 rate limit，已将对应的 current_streak 置为上限。")
            else:
                # 常规释放流程
                manager.release_account(account_name, error_detail)
        except Exception as e:
            # 避免释放流程抛出异常导致上层失败，记录并尝试常规释放一次
            print(f"{log_prefix} 在处理释放/rate-limit 更新时发生异常: {e}")
            try:
                manager.release_account(account_name, error_detail)
            except Exception as e2:
                print(f"{log_prefix} 尝试常规释放账号时再次失败: {e2}")

    return error_detail, result_text


def validate_all_accounts():
    """
    遍历配置文件中的所有账号，并测试其可用性。
    """
    print("--- 开始验证所有账号有效性 ---")

    # 转换为字符串路径使用 read_json
    config_path_str = str(CONFIG_FILE)

    # 使用 read_json 读取配置
    config_data = read_json(config_path_str)

    # 检查是否读取到数据
    if not config_data:
        print(f"错误: 配置文件 {config_path_str} 不存在或内容为空/格式错误。")
        return

    account_list = config_data.get("account_list", [])
    if not account_list:
        print("配置文件中没有找到任何账号 (account_list 为空)。")
        return

    valid_accounts = []
    invalid_accounts = []
    total_accounts = len(account_list)

    for i, account in enumerate(account_list):
        name = account.get("name")
        user_data_dir = account.get("user_data_dir")

        if not name or not user_data_dir:
            print(f"跳过配置不完整的条目: {account}")
            total_accounts -= 1
            continue

        print(f"\n[{i + 1}/{total_accounts}] 正在验证账号: {name}...")
        test_file = r"W:\project\python_project\watermark_remove\common_utils\video_scene\test.jpg"

        # 直接调用 query_google_ai_studio 进行测试
        error, response = query_google_ai_studio(
            prompt="你是谁",
            file_path=test_file,
            user_data_dir=user_data_dir
        )

        if error is None and response and response.strip():
            # 这里的打印也可以顺便截取一下，保持日志整洁
            print(f"  [✅ OK] 账号 {name} 验证通过。")

            valid_accounts.append({
                "name": name,
                "response": response
            })
        else:
            error_reason = str(error)[:250].replace('\n', ' ') + "..." if error else "模型返回为空"
            print(f"  [❌ 失效] 账号 {name} 验证失败。")
            invalid_accounts.append({"name": name, "reason": error_reason})

    # 打印总结报告
    print("\n" + "=" * 50)
    print("          账号验证总结")
    print("=" * 50)
    print(f"总计账号: {total_accounts}")
    print(f"有效账号: {len(valid_accounts)}")
    print(f"失效账号: {len(invalid_accounts)}")

    if valid_accounts:
        print("\n--- ✅ 有效账号列表 ---")
        for item in valid_accounts:
            raw_response = item['response'].strip().replace('\n', ' ').replace('\r', ' ')
            preview = raw_response[:50]
            if len(raw_response) > 50:
                preview += "..."

            print(f"- {item['name']}: {preview}")

    if invalid_accounts:
        print("\n--- ❌ 失效账号列表及原因 ---")
        for item in invalid_accounts:
            print(f"- {item['name']}: {item['reason']}")
    print("=" * 50)


if __name__ == "__main__":
    validate_all_accounts()
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

class PlaywrightAccountManager:
    def __init__(self, config_path, stats_path):
        self.config_path = config_path
        self.stats_path = stats_path
        self.lock_path = str(stats_path) + ".lock"

    def _check_and_reset_stuck_accounts(self, stats_data, timeout_seconds=900):
        """检查并重置长时间处于 using 状态的账号"""
        now = datetime.now()
        # 【修改点】只检查账号信息，不检查顶级字段
        for name, info in stats_data.items():
            # 跳过非字典类型的顶级键 (如我们新增的 active_pool)
            if not isinstance(info, dict):
                continue
            if info.get('status') == 'using':
                last_time_str = info.get('last_used_time', '')
                if last_time_str:
                    try:
                        last_time = datetime.strptime(last_time_str, "%Y-%m-%d %H:%M:%S")
                        if (now - last_time).total_seconds() > timeout_seconds:
                            print(f"[Manager] 账号 {name} 'using'状态超时({timeout_seconds}s)，强制重置为'idle'")
                            info['status'] = 'idle'
                            info['last_error_info'] = "System: Force reset due to timeout"
                    except ValueError:
                        info['status'] = 'idle'
                        info['last_error_info'] = "System: Force reset due to invalid time format"
        return stats_data

    def allocate_account(self, model_name=None):
        """
        分配一个空闲账号。
        【修复版】引入前置全局并发检查，防止轮换瞬间产生 N+1 个进程。
        """
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

            # 3.1 基础同步与清理
            for name in list(stats.keys()):
                if name == 'active_pool': continue
                if name not in valid_accounts_map: del stats[name]

            for name in valid_accounts_map:
                if name not in stats:
                    stats[name] = {"status": "idle", "last_used_time": "", "last_error_info": None, "total_usage": 0,
                                   "current_streak": 0, "last_used_model": None}

            # 3.2 超时重置
            stats = self._check_and_reset_stuck_accounts(stats)

            # 4. 【核心修复：前置全局并发检查】
            # 必须先统计全系统正在 status='using' 的总数，达到上限直接返回 None，不执行后续轮换逻辑
            current_using_total = sum(
                1 for name, info in stats.items() if isinstance(info, dict) and info.get('status') == 'using')
            if current_using_total >= max_concurrency:
                save_json(self.stats_path, stats)
                return None, None

            # 5. 维护活跃池 (Active Pool)
            exhausted_accounts = set()
            for name, info in stats.items():
                if isinstance(info, dict) and info.get('current_streak', 0) >= usage_streak_limit:
                    info['current_streak'] = 0  # 达到上限，准备轮换
                    exhausted_accounts.add(name)

            active_pool = stats.get('active_pool', [])
            # 过滤掉已失效或已用完次数的账号
            active_pool = [n for n in active_pool if n in valid_accounts_map and n not in exhausted_accounts]

            # 如果池不满，补充新账号（优先选总使用次数最少的，且必须满足30分钟冷却时间）
            needed = max_concurrency - len(active_pool)
            if needed > 0:
                # 定义冷却时间 30分钟
                COOLDOWN_SECONDS = 1200
                now_dt = datetime.now()

                def is_cooldown_ready(acc_name):
                    """检查账号上次使用时间是否在30分钟前"""
                    last_time_str = stats.get(acc_name, {}).get('last_used_time', '')
                    if not last_time_str:
                        return True  # 从未使用的账号，直接可用
                    try:
                        last_time = datetime.strptime(last_time_str, "%Y-%m-%d %H:%M:%S")
                        # 只有当 (当前时间 - 上次时间) > 30分钟 时才允许被选中进入活跃池
                        return (now_dt - last_time).total_seconds() > COOLDOWN_SECONDS
                    except ValueError:
                        return True

                # 筛选候选人：在有效列表中 + 不在当前活跃池中 + 满足冷却时间
                candidates = [
                    n for n in valid_accounts_map
                    if n not in active_pool
                       and is_cooldown_ready(n)
                ]

                # 按总使用次数排序，优先使用少的
                candidates.sort(key=lambda n: stats.get(n, {}).get('total_usage', 0))
                active_pool.extend(candidates[:needed])

            stats['active_pool'] = active_pool

            # 6. 从活跃池中选择一个空闲账号分配
            target_name = None
            idle_in_pool = [n for n in active_pool if stats.get(n, {}).get('status') == 'idle']

            if idle_in_pool:
                idle_in_pool.sort(key=lambda n: stats.get(n, {}).get('total_usage', 0))
                target_name = idle_in_pool[0]
            # else:
            #     # 备选逻辑：如果池内恰好没空位（极其罕见），从池外找一个空闲的
            #     all_idle = [n for n, info in stats.items() if isinstance(info, dict) and info.get('status') == 'idle']
            #     if all_idle:
            #         all_idle.sort(key=lambda n: stats.get(n, {}).get('total_usage', 0))
            #         target_name = all_idle[0]

            if not target_name:
                save_json(self.stats_path, stats)
                return None, None

            # 7. 更新状态
            target_info = stats[target_name]
            target_info['status'] = 'using'
            target_info['last_used_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            target_info['total_usage'] = target_info.get('total_usage', 0) + 1
            target_info['current_streak'] = target_info.get('current_streak', 0) + 1
            target_info['last_used_model'] = model_name
            save_json(self.stats_path, stats)
            return target_name, valid_accounts_map[target_name]

    def release_account(self, account_name, error_info=None):
        """释放账号，重置为 idle。注意：此函数不重置 streak，streak 只在分配时管理。"""
        with SimpleFileLock(self.lock_path):
            stats = read_json(self.stats_path)

            if account_name in stats and isinstance(stats[account_name], dict):
                info = stats[account_name]
                info['status'] = 'idle'
                info['last_used_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                # 记录最后一次错误信息，如果成功则为 None
                info['last_error_info'] = str(error_info)[:1000] if error_info else None
                # 如果有错误，可以选择重置连续使用计数，让它有机会被轮换出去
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

    no_file_name_list = ['dahao', 'new_taobao14']
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

        # 如果返回包含 rate limit 的提示，把该账号的 current_streak 直接提升到上限，
        # 并相应增加 total_usage，防止短时间内再次选到它。
        try:
            if result_text and "You've reached your rate limit. Please try again later" in result_text:
                with SimpleFileLock(manager.lock_path):
                    stats = read_json(manager.stats_path)
                    raw_config = read_json(manager.config_path) or {}
                    usage_streak_limit = raw_config.get('usage_streak_limit', DEFAULT_USAGE_STREAK_LIMIT)

                    if account_name in stats and isinstance(stats[account_name], dict):
                        info = stats[account_name]
                        cur = info.get('current_streak', 0)
                        if cur < usage_streak_limit:
                            inc = usage_streak_limit - cur
                            info['current_streak'] = usage_streak_limit
                            info['total_usage'] = info.get('total_usage', 0) + inc
                        # 标记为 idle，记录最后使用时间与错误信息，避免后续 release_account 把 streak 重置
                        info['status'] = 'idle'
                        info['last_used_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        info['last_error_info'] = "Remote: rate limit detected"
                    save_json(manager.stats_path, stats)
                print(
                    f"{log_prefix} 账号 {account_name} 检测到 rate limit，已将 current_streak 置为上限并增加 total_usage。")
            else:
                # 常规释放流程（包含可能的 error_detail，会在 release_account 中记录并在有错误时重置 streak）
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
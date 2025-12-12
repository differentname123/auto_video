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

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_FILE = BASE_DIR / 'config/gemini_auto.json'
STATS_FILE = BASE_DIR / 'config/gemini_auto_stats.json'
# 假设这个引用是存在的，保持不变
from utils.auto_web.web_auto import query_google_ai_studio
from utils.common_utils import save_json, read_json


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
        for name, info in stats_data.items():
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

    def allocate_account(self):
        """分配一个空闲账号"""
        with SimpleFileLock(self.lock_path):
            # 读取配置和统计信息
            raw_config = read_json(self.config_path)
            stats = read_json(self.stats_path)

            config_list = raw_config.get('account_list', [])

            # 建立有效账号映射表
            valid_accounts_map = {
                item['name']: item.get('user_data_dir', '')
                for item in config_list if item.get('name') and item.get('user_data_dir')
            }

            # 1. 清理：从统计中移除已在配置中删除的账号
            for name in list(stats.keys()):
                if name not in valid_accounts_map:
                    print(f"[Manager] 从配置中移除账号 {name}，同步删除统计信息。")
                    del stats[name]

            # 2. 初始化：为新账号添加统计记录
            for name in valid_accounts_map:
                if name not in stats:
                    stats[name] = {
                        "status": "idle",
                        "last_used_time": "",
                        "last_error_info": None,
                        "total_usage": 0
                    }

            # 3. 检查异常状态
            stats = self._check_and_reset_stuck_accounts(stats)

            # 4. 筛选空闲账号
            candidates = [
                {'name': name, 'count': info.get('total_usage', 0)}
                for name, info in stats.items() if info.get('status') == 'idle'
            ]

            if not candidates:
                # 没有可用账号，保存当前状态（可能发生了重置）并返回
                save_json(self.stats_path, stats)
                return None, None

            # 5. 负载均衡策略：随机打乱后，选使用次数最少的
            random.shuffle(candidates)
            best_account = sorted(candidates, key=lambda x: x['count'])[0]
            target_name = best_account['name']

            # 6. 更新状态为 using
            target_info = stats[target_name]
            target_info['status'] = 'using'
            target_info['last_used_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            target_info['total_usage'] = target_info.get('total_usage', 0) + 1

            save_json(self.stats_path, stats)

            return target_name, valid_accounts_map[target_name]

    def release_account(self, account_name, error_info=None):
        """释放账号，重置为 idle"""
        with SimpleFileLock(self.lock_path):
            stats = read_json(self.stats_path)

            if account_name in stats:
                info = stats[account_name]
                info['status'] = 'idle'
                info['last_used_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                # 记录最后一次错误信息，如果成功则为 None
                info['last_error_info'] = str(error_info)[:1000] if error_info else None

            save_json(self.stats_path, stats)


# ==========================================
# 4. 对外统一接口
# ==========================================


# 确保配置目录存在
if not CONFIG_FILE.parent.exists():
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)

manager = PlaywrightAccountManager(str(CONFIG_FILE), str(STATS_FILE))


def generate_gemini_content_playwright(prompt, files=None, wait_timeout=600):
    """
    使用 Playwright 账号管理器安全地调用 Gemini。
    """
    pid = os.getpid()
    tid = threading.get_ident()
    log_prefix = f"[System][PID:{pid},TID:{tid}]"

    start_time = time.time()
    account_name, user_data_dir = None, None

    # 1. 循环申请账号
    while time.time() - start_time < wait_timeout:
        account_name, user_data_dir = manager.allocate_account()
        if account_name:
            break

        elapsed = int(time.time() - start_time)
        print(f"{log_prefix} 无可用账号，进入等待... (已等待 {elapsed}s / {wait_timeout}s)")
        time.sleep(random.uniform(5, 15))

    if not account_name:
        return f"System Busy: 等待 {wait_timeout} 秒后仍无可用账号。", None

    print(f"{log_prefix} 分配账号: {account_name} ({os.path.basename(user_data_dir)})")

    error_detail, result_text = None, None
    try:
        file_to_upload = None
        if files:
            if isinstance(files, list) and files:
                file_to_upload = files[0]
            elif isinstance(files, str):
                file_to_upload = files

        # 2. 调用核心函数
        error_detail, result_text = query_google_ai_studio(
            prompt=prompt,
            file_path=file_to_upload,
            user_data_dir=user_data_dir
        )
    except Exception as e:
        error_detail = f"管理器外部发生严重错误: {str(e)}\n\n{traceback.format_exc()}"
    finally:
        # 3. 释放账号
        print(f"{log_prefix} 释放账号: {account_name}")
        manager.release_account(account_name, error_detail)

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

        # 直接调用 query_google_ai_studio 进行测试
        error, response = query_google_ai_studio(
            prompt="你是谁",
            user_data_dir=user_data_dir
        )

        if error is None and response and response.strip():
            # 这里的打印也可以顺便截取一下，保持日志整洁
            print(f"  [✅ OK] 账号 {name} 验证通过。")

            # 【修改点 1】: 将 name 和 response 一起存入 valid_accounts
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
            # 【修改点 2】: 提取响应，去除换行符，并截取前50个字符
            raw_response = item['response'].strip().replace('\n', ' ').replace('\r', ' ')
            preview = raw_response[:50]
            # 如果长度超过50，加省略号
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
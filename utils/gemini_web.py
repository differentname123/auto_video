import asyncio
import json
import os
import time
import re
import traceback
import random
from datetime import datetime
from pathlib import Path

from gemini_webapi import GeminiClient, set_log_level
from gemini_webapi.exceptions import AuthError
from utils.common_utils import read_json

# 设置日志级别
set_log_level("INFO")


# ==========================================
# 1. 基础工具类：文件锁 (保持不变)
# ==========================================

class SimpleFileLock:
    def __init__(self, lock_file, timeout=10):
        self.lock_file = lock_file
        self.timeout = timeout

    def __enter__(self):
        start_time = time.time()
        while True:
            try:
                self.fd = os.open(self.lock_file, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                return self
            except FileExistsError:
                if time.time() - start_time > self.timeout:
                    raise TimeoutError(f"获取文件锁超时: {self.lock_file}")
                time.sleep(0.1)

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            os.close(self.fd)
            os.remove(self.lock_file)
        except OSError:
            pass


# ==========================================
# 2. 核心修改：账号管理器
# ==========================================

class GeminiAccountManager:
    def __init__(self, config_path, stats_path):
        self.config_path = config_path
        self.stats_path = stats_path
        self.lock_path = str(stats_path) + ".lock"

    def _read_json_safe(self, path):
        if not os.path.exists(path):
            return {}
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                return json.loads(content) if content else {}
        except Exception:
            return {}

    def _check_and_reset_stuck_accounts(self, stats_data, timeout_seconds=300):
        """检查并重置僵死账号"""
        now = datetime.now()
        for name, info in stats_data.items():
            if info.get('status') == 'using':
                last_time_str = info.get('last_used_time', '')
                if last_time_str:
                    try:
                        last_time = datetime.strptime(last_time_str, "%Y-%m-%d %H:%M:%S")
                        if (now - last_time).total_seconds() > timeout_seconds:
                            print(f"[Manager] 账号 {name} 超时，强制重置")
                            info['status'] = 'idle'
                            info['last_error_info'] = "System: Force reset due to timeout"
                    except ValueError:
                        pass
        return stats_data

    def allocate_account(self, model_name):
        """
        申请账号：
        1. 同步 Config 与 Stats (删除废弃账号，添加新账号)。
        2. 找到 idle 账号。
        3. 立即增加调用计数。
        4. 返回 name 和 cookie (cookie 来自 config)。
        """
        with SimpleFileLock(self.lock_path):
            # 1. 读取 Config (作为 Cookie 和账号存在性的权威来源)
            raw_config = self._read_json_safe(self.config_path)
            config_list = raw_config.get('cookie_list', [])

            # 构建 config 映射: name -> cookie
            valid_accounts_map = {
                item['name']: item.get('cookie_str', '')
                for item in config_list if item.get('name')
            }

            # 2. 读取 Stats
            stats = self._read_json_safe(self.stats_path)

            # 3. [关键调整] 同步逻辑：Stats 必须完全匹配 Config 的 key

            # 3.1 删除：Config 里没有的，Stats 里也要删掉
            current_stats_keys = list(stats.keys())
            for name in current_stats_keys:
                if name not in valid_accounts_map:
                    print(f"[Manager] 检测到账号 {name} 已从配置中移除，同步删除统计信息。")
                    del stats[name]

            # 3.2 新增：Config 里有的，Stats 里没有的，初始化
            for name in valid_accounts_map:
                if name not in stats:
                    stats[name] = {
                        # 注意：这里不再存储 cookie_str
                        "status": "idle",
                        "last_used_time": "",
                        "last_error_info": None,
                        "model_usage": {}
                    }

            # 4. 清理僵死状态
            stats = self._check_and_reset_stuck_accounts(stats)

            # 5. 筛选可用账号
            candidates = []
            for name, info in stats.items():
                if info.get('status') == 'idle':
                    count = info.get('model_usage', {}).get(model_name, {}).get('count', 0)
                    candidates.append({
                        'name': name,
                        'count': count
                    })

            if not candidates:
                # 即使没有可用账号，也要保存同步后的状态（比如删除了账号）
                with open(self.stats_path, 'w', encoding='utf-8') as f:
                    json.dump(stats, f, indent=4, ensure_ascii=False)
                return None, None

            # 6. 排序与选择
            random.shuffle(candidates)
            best_account = sorted(candidates, key=lambda x: x['count'])[0]
            target_name = best_account['name']

            # 7. [关键调整] 状态更新：立即增加计数，不再等 release
            target_info = stats[target_name]
            target_info['status'] = 'using'
            target_info['last_used_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if 'model_usage' not in target_info:
                target_info['model_usage'] = {}
            if model_name not in target_info['model_usage']:
                target_info['model_usage'][model_name] = {'count': 0}

            # 立即 +1
            target_info['model_usage'][model_name]['count'] += 1

            # 8. 写回 Stats (不含 cookie)
            with open(self.stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=4, ensure_ascii=False)

            # 9. 返回 Name 和 Cookie (从 Config 字典中取)
            return target_name, valid_accounts_map[target_name]

    def release_account(self, account_name, error_info=None):
        """
        释放账号：
        只更新状态和错误信息，不处理计数（因为 allocate 时已经加过了）。
        """
        with SimpleFileLock(self.lock_path):
            stats = self._read_json_safe(self.stats_path)

            if account_name in stats:
                info = stats[account_name]
                info['status'] = 'idle'
                info['last_used_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 更新为结束时间

                if error_info:
                    info['last_error_info'] = str(error_info)[0:500]
                else:
                    info['last_error_info'] = None

            with open(self.stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=4, ensure_ascii=False)


# ==========================================
# 3. 异步请求逻辑
# ==========================================

async def _do_gemini_request(cookie_str, prompt, model_name, files):
    PROXY_URL = "http://127.0.0.1:7890"

    def _get_val(k, t):
        m = re.search(f"{re.escape(k)}=([^;]+)", t)
        return m.group(1) if m else None

    psid = _get_val("__Secure-1PSID", cookie_str)
    psidts = _get_val("__Secure-1PSIDTS", cookie_str)

    if not psid:
        raise ValueError("Cookie 无效：缺少 __Secure-1PSID")

    client = GeminiClient(psid, psidts, proxy=PROXY_URL)

    try:
        await client.init(timeout=600, auto_close=False, auto_refresh=True)
        response = await client.generate_content(prompt, model=model_name, files=files)
        return response.text
    except Exception as e:
        raise e


# ==========================================
# 4. 对外接口
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_FILE = BASE_DIR / 'config/gemini_web.json'
STATS_FILE = BASE_DIR / 'config/gemini_web_stats.json'

manager = GeminiAccountManager(str(CONFIG_FILE), str(STATS_FILE))


def generate_gemini_content_managed(prompt, model_name="gemini-2.5-pro", files=None):
    """
    对外提供的统一接口
    """
    # 1. 申请账号 (此时计数已+1)
    account_name, cookie = manager.allocate_account(model_name)

    if not account_name:
        return "System Busy: 无可用账号或配置为空", None

    print(f"[System] 分配账号: {account_name}")

    error_detail = None
    result_text = None

    try:
        # 2. 执行请求
        result_text = asyncio.run(_do_gemini_request(cookie, prompt, model_name, files))
    except Exception as e:
        error_detail = f"{str(e)}\nTraceback: {traceback.format_exc()}"
    finally:
        # 3. 释放账号 (仅重置状态)
        manager.release_account(account_name, error_detail)

    return error_detail, result_text


# ==========================================
# 测试部分
# ==========================================

if __name__ == "__main__":

    print("开始测试...")

    # 模拟 Config 文件里必须有数据才能跑
    # 运行后请检查 gemini_stats.json，里面应该只有 counts 和 status，没有 cookie

    err, res = generate_gemini_content_managed(
        prompt="你是哪个型号",
        model_name="gemini-2.5-pro"
    )

    if err:
        print(f"失败: {err}")
    else:
        print(f"成功: {res}")
import requests
import concurrent.futures
import time

from utils.common_utils import read_json, save_json

# ================= 配置区 =================
# 您的本地翻墙代理（用于拉取被墙的代理源，如 ProxyScrape 或 GitHub Raw）
LOCAL_PROXY = {
    "http": "http://127.0.0.1:7890",
    "https": "http://127.0.0.1:7890"
}

# 测试代理存活的目标网站（B站基础接口）
TEST_TARGET_URL = "https://api.bilibili.com/x/web-interface/nav"

# 基础超时宽容度（秒）：超过这个时间的代理直接判定死亡，不再参与排序
BASE_TIMEOUT = 10.0  # 建议稍微调低到10秒，提高筛选效率

# 代理状态保存路径
PROXY_STATUS_FILE = r"W:\project\python_project\auto_video\config\proxy_status.json"


# ==========================================
# 代理源拉取函数区 (每个函数独立，增加了最多5次重试机制)
# ==========================================

def fetch_from_proxyscrape():
    """从 ProxyScrape 接口拉取代理"""
    print("正在拉取源: ProxyScrape...")
    url = "https://api.proxyscrape.com/v4/free-proxy-list/get?request=displayproxies&protocol=http&anonymity=elite,anonymous"
    for attempt in range(5):
        try:
            response = requests.get(url, proxies=LOCAL_PROXY, timeout=15)
            if response.status_code == 200:
                proxies = [p.strip() for p in response.text.split('\n') if p.strip()]
                print(f"  └─ 成功获取 {len(proxies)} 个")
                return proxies
        except Exception as e:
            print(f"  └─ ❌ 获取失败 (第 {attempt + 1}/5 次重试): {e}")
            if attempt < 4:
                time.sleep(1)  # 失败后短暂休眠1秒再重试
    return []


def fetch_from_github_thespeedx():
    """从 GitHub: TheSpeedX/PROXY-List 拉取 HTTP 代理"""
    print("正在拉取源: GitHub (TheSpeedX)...")
    url = "https://raw.githubusercontent.com/TheSpeedX/PROXY-List/master/http.txt"
    for attempt in range(5):
        try:
            response = requests.get(url, proxies=LOCAL_PROXY, timeout=15)
            if response.status_code == 200:
                proxies = [p.strip() for p in response.text.split('\n') if p.strip()]
                print(f"  └─ 成功获取 {len(proxies)} 个")
                return proxies
        except Exception as e:
            print(f"  └─ ❌ 获取失败 (第 {attempt + 1}/5 次重试): {e}")
            if attempt < 4:
                time.sleep(1)
    return []


def fetch_from_github_monosans():
    """从 GitHub: monosans/proxy-list 拉取 HTTP 代理"""
    print("正在拉取源: GitHub (monosans)...")
    url = "https://raw.githubusercontent.com/monosans/proxy-list/main/proxies/http.txt"
    for attempt in range(5):
        try:
            response = requests.get(url, proxies=LOCAL_PROXY, timeout=15)
            if response.status_code == 200:
                proxies = [p.strip() for p in response.text.split('\n') if p.strip()]
                print(f"  └─ 成功获取 {len(proxies)} 个")
                return proxies
        except Exception as e:
            print(f"  └─ ❌ 获取失败 (第 {attempt + 1}/5 次重试): {e}")
            if attempt < 4:
                time.sleep(1)
    return []


def fetch_from_github_proxifly():
    """从 GitHub: proxifly/free-proxy-list 拉取 HTTP 代理"""
    print("正在拉取源: GitHub (proxifly)...")
    url = "https://raw.githubusercontent.com/proxifly/free-proxy-list/main/proxies/protocols/http/data.txt"
    for attempt in range(5):
        try:
            # proxifly 的格式包含了其他信息，通常代理在第一列，用逗号或空格分隔
            response = requests.get(url, proxies=LOCAL_PROXY, timeout=15)
            if response.status_code == 200:
                lines = [p.strip() for p in response.text.split('\n') if p.strip()]
                proxies = []
                for line in lines:
                    raw_proxy = line.split()[0].split(',')[0]
                    # 核心修复：去除自带的 http:// 或 https:// 前缀，防止下游拼接错乱
                    clean_proxy = raw_proxy.replace("http://", "").replace("https://", "")
                    proxies.append(clean_proxy)
                print(f"  └─ 成功获取 {len(proxies)} 个")
                return proxies
        except Exception as e:
            print(f"  └─ ❌ 获取失败 (第 {attempt + 1}/5 次重试): {e}")
            if attempt < 4:
                time.sleep(1)
    return []

# === 以下为新增的 5 个高质量源 ===

def fetch_from_github_shiftytr():
    """从 GitHub: ShiftyTR/Proxy-List 拉取 HTTP 代理"""
    print("正在拉取源: GitHub (ShiftyTR)...")
    url = "https://raw.githubusercontent.com/ShiftyTR/Proxy-List/master/http.txt"
    for attempt in range(5):
        try:
            response = requests.get(url, proxies=LOCAL_PROXY, timeout=15)
            if response.status_code == 200:
                proxies = [p.strip() for p in response.text.split('\n') if p.strip()]
                print(f"  └─ 成功获取 {len(proxies)} 个")
                return proxies
        except Exception as e:
            print(f"  └─ ❌ 获取失败 (第 {attempt + 1}/5 次重试): {e}")
            if attempt < 4:
                time.sleep(1)
    return []


def fetch_from_github_jetkai():
    """从 GitHub: jetkai/proxy-list 拉取 HTTP 代理"""
    print("正在拉取源: GitHub (jetkai)...")
    url = "https://raw.githubusercontent.com/jetkai/proxy-list/main/online-proxies/txt/proxies-http.txt"
    for attempt in range(5):
        try:
            response = requests.get(url, proxies=LOCAL_PROXY, timeout=15)
            if response.status_code == 200:
                proxies = [p.strip() for p in response.text.split('\n') if p.strip()]
                print(f"  └─ 成功获取 {len(proxies)} 个")
                return proxies
        except Exception as e:
            print(f"  └─ ❌ 获取失败 (第 {attempt + 1}/5 次重试): {e}")
            if attempt < 4:
                time.sleep(1)
    return []



def fetch_from_spys_me():
    """从 spys.me 拉取 HTTP 代理"""
    print("正在拉取源: spys.me...")
    url = "https://spys.me/proxy.txt"
    for attempt in range(5):
        try:
            response = requests.get(url, proxies=LOCAL_PROXY, timeout=15)
            if response.status_code == 200:
                lines = [p.strip() for p in response.text.split('\n') if p.strip()]
                proxies = []
                # spys.me 的格式包含说明和附加信息，需要过滤出有效的 IP:Port
                for line in lines:
                    parts = line.split()
                    if parts and ":" in parts[0] and parts[0][0].isdigit():
                        proxies.append(parts[0])
                print(f"  └─ 成功获取 {len(proxies)} 个")
                return proxies
        except Exception as e:
            print(f"  └─ ❌ 获取失败 (第 {attempt + 1}/5 次重试): {e}")
            if attempt < 4:
                time.sleep(1)
    return []

# ==========================================
# 测速与调度核心代码
# ==========================================

def format_and_test_proxy(proxy_address):
    """
    测试并返回：(代理地址字符串, 代理字典, 响应时间)
    如果不通，响应时间默认返回 100.0
    """
    proxy_dict = {
        "http": f"http://{proxy_address}",
        "https": f"http://{proxy_address}"
    }

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    }

    try:
        response = requests.get(
            TEST_TARGET_URL,
            proxies=proxy_dict,
            headers=headers,
            timeout=BASE_TIMEOUT
        )
        if response.status_code == 200 and "code" in response.json():
            elapsed_time = response.elapsed.total_seconds()
            return (proxy_address, proxy_dict, elapsed_time)
    except:
        pass

    # 不通时默认延迟 100
    return (proxy_address, proxy_dict, 100.0)


def fetch_from_github_clarketm():
    """从 GitHub: clarketm/proxy-list 拉取代理（每日更新）"""
    print("正在拉取源: GitHub (clarketm)...")
    url = "https://raw.githubusercontent.com/clarketm/proxy-list/master/proxy-list-raw.txt"
    for attempt in range(5):
        try:
            response = requests.get(url, proxies=LOCAL_PROXY, timeout=15)
            if response.status_code == 200:
                proxies = [p.strip() for p in response.text.split('\n') if p.strip()]
                print(f"  └─ 成功获取 {len(proxies)} 个")
                return proxies
        except Exception as e:
            print(f"  └─ ❌ 获取失败 (第 {attempt + 1}/5 次重试): {e}")
            if attempt < 4:
                time.sleep(1)
    return []


def fetch_from_github_komutan234():
    """从 GitHub: komutan234/Proxy-List-Free 拉取 HTTP 代理（每2分钟更新）"""
    print("正在拉取源: GitHub (komutan234)...")
    url = "https://raw.githubusercontent.com/komutan234/Proxy-List-Free/main/proxies/http.txt"
    for attempt in range(5):
        try:
            response = requests.get(url, proxies=LOCAL_PROXY, timeout=15)
            if response.status_code == 200:
                proxies = [p.strip() for p in response.text.split('\n') if p.strip()]
                print(f"  └─ 成功获取 {len(proxies)} 个")
                return proxies
        except Exception as e:
            print(f"  └─ ❌ 获取失败 (第 {attempt + 1}/5 次重试): {e}")
            if attempt < 4:
                time.sleep(1)
    return []


def fetch_from_github_iplocate():
    """从 GitHub: iplocate/free-proxy-list 拉取 HTTP 代理（每30分钟更新）"""
    print("正在拉取源: GitHub (iplocate)...")
    url = "https://raw.githubusercontent.com/iplocate/free-proxy-list/main/protocols/http.txt"
    for attempt in range(5):
        try:
            response = requests.get(url, proxies=LOCAL_PROXY, timeout=15)
            if response.status_code == 200:
                proxies = [p.strip() for p in response.text.split('\n') if p.strip()]
                print(f"  └─ 成功获取 {len(proxies)} 个")
                return proxies
        except Exception as e:
            print(f"  └─ ❌ 获取失败 (第 {attempt + 1}/5 次重试): {e}")
            if attempt < 4:
                time.sleep(1)
    return []


def fetch_from_github_mmpx12():
    """从 GitHub: mmpx12/proxy-list 拉取 HTTP 代理（每小时更新）"""
    print("正在拉取源: GitHub (mmpx12)...")
    url = "https://raw.githubusercontent.com/mmpx12/proxy-list/master/http.txt"
    for attempt in range(5):
        try:
            response = requests.get(url, proxies=LOCAL_PROXY, timeout=15)
            if response.status_code == 200:
                proxies = [p.strip() for p in response.text.split('\n') if p.strip()]
                print(f"  └─ 成功获取 {len(proxies)} 个")
                return proxies
        except Exception as e:
            print(f"  └─ ❌ 获取失败 (第 {attempt + 1}/5 次重试): {e}")
            if attempt < 4:
                time.sleep(1)
    return []



def fetch_from_github_zloi():
    """GitHub: zloi-user/hideip.me"""
    print("正在拉取源: GitHub (zloi-user)...")
    url = "https://raw.githubusercontent.com/zloi-user/hideip.me/main/http.txt"
    for attempt in range(5):
        try:
            response = requests.get(url, proxies=LOCAL_PROXY, timeout=15)
            if response.status_code == 200:
                lines = [p.strip() for p in response.text.split('\n') if p.strip()]
                proxies = []
                for line in lines:
                    # 核心修复：格式通常为 ip:port:country，以冒号分割并严格截取前两部分组装
                    parts = line.split(':')
                    if len(parts) >= 2:
                        proxies.append(f"{parts[0]}:{parts[1]}")
                print(f"  └─ 成功获取 {len(proxies)} 个")
                return proxies
        except Exception as e:
            print(f"  └─ ❌ 获取失败 (第 {attempt + 1}/5 次重试): {e}")
            if attempt < 4:
                time.sleep(1)
    return []

def fetch_from_github_prxchk():
    """GitHub: prxchk/proxy-list"""
    print("正在拉取源: GitHub (prxchk)...")
    url = "https://raw.githubusercontent.com/prxchk/proxy-list/main/http.txt"
    for attempt in range(5):
        try:
            response = requests.get(url, proxies=LOCAL_PROXY, timeout=15)
            if response.status_code == 200:
                proxies = [p.strip() for p in response.text.split('\n') if p.strip()]
                print(f"  └─ 成功获取 {len(proxies)} 个")
                return proxies
        except Exception as e:
            print(f"  └─ ❌ 获取失败 (第 {attempt + 1}/5 次重试): {e}")
            if attempt < 4:
                time.sleep(1)
    return []


def get_top_proxies(count=10):
    """
    主控函数：多源拉取 -> 去重 -> 多线程测速 -> 记录历史得分 -> 排序 -> 提取 Top N
    """
    print("=== 第一阶段：从各个数据源聚合代理 ===")

    # 【灵活启用/关闭模块】：将不想用的函数注释掉即可
    sources = [
        fetch_from_proxyscrape,
        fetch_from_github_thespeedx,
        fetch_from_github_monosans,
        fetch_from_github_proxifly,


        # 挂载新增的数据源 gemini
        fetch_from_github_shiftytr,
        fetch_from_github_jetkai,
        fetch_from_spys_me,


        # grok
        fetch_from_github_clarketm,
        fetch_from_github_komutan234,
        fetch_from_github_iplocate,
        fetch_from_github_mmpx12,

        # gpt
        fetch_from_github_zloi,
        fetch_from_github_prxchk
    ]

    raw_proxies = []
    for source_func in sources:
        proxy_list = source_func()
        raw_proxies.extend(proxy_list)
        # 打印一个proxy_list的元素
        if proxy_list:
            print(f"  └─ 示例: {proxy_list[0]} (共 {len(proxy_list)} 个) 函数名: {source_func.__name__}")

    # 列表去重（不同源之间可能有大量重复的公共免费IP）
    unique_proxies = list(set(raw_proxies))
    print(f"\n🔄 聚合完毕，去重后共获得 {len(unique_proxies)} 个独立代理 IP。")

    if not unique_proxies:
        print("❌ 没有任何原始代理，退出。")
        return []

    print(f"\n=== 第二阶段：并发测速 (宽容超时: {BASE_TIMEOUT}秒) ===")

    # 载入现有的代理状态记录
    proxy_status = read_json(PROXY_STATUS_FILE)
    if not isinstance(proxy_status, dict):
        proxy_status = {}

    current_time = time.time()
    valid_proxies_with_time = []

    # 由于 GitHub 源加起来可能有上万个 IP，建议把最大线程数拉高到 300-500 以加快速度
    max_threads = min(500, len(unique_proxies))
    print(f"🚀 启动 {max_threads} 个线程进行轰炸式测速...")

    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
        results = executor.map(format_and_test_proxy, unique_proxies)

        for res in results:
            proxy_str, p_dict, delay = res

            # 初始化代理字典结构（兼容旧版本数据）
            if proxy_str not in proxy_status:
                proxy_status[proxy_str] = {"history_list": []}
            elif "history_list" not in proxy_status[proxy_str]:
                proxy_status[proxy_str]["history_list"] = []

            # 1. 添加本次测速记录
            proxy_status[proxy_str]["history_list"].append({
                "time": current_time,
                "delay": delay
            })

            # 2. 清理历史：保留最近 1 小时的记录
            history_list = proxy_status[proxy_str]["history_list"]
            history_list = [record for record in history_list if current_time - record["time"] <= 3600]

            # 保留最近的 10 个测试记录
            history_list = history_list[-10:]

            # 3. 计算平均延迟作为最终得分
            if history_list:
                avg_delay = sum(record["delay"] for record in history_list) / len(history_list)
            else:
                avg_delay = 100.0

            # 4. 重新赋值更新，确保 delay 和 is_high_anon 字段在 history_list 前面
            existing_anon = proxy_status[proxy_str].get("is_high_anon")
            new_status_data = {"delay": round(avg_delay, 3)}

            # 防覆盖机制：如果有已测过的高匿标识，原样保留
            if existing_anon is not None:
                new_status_data["is_high_anon"] = existing_anon

            new_status_data["history_list"] = history_list
            proxy_status[proxy_str] = new_status_data

            # 我们只把平均延迟低于100（说明近期有过至少一次通的记录）的代理放入备选列表
            if avg_delay < 50.0:
                valid_proxies_with_time.append((p_dict, avg_delay))

    # 执行完毕后，处理要保存的 JSON 数据
    # 将整个代理字典按照 delay 升序排列，并强制让所有节点的 delay 字段都排在前面
    sorted_proxy_status = {}
    sorted_items = sorted(proxy_status.items(), key=lambda x: x[1].get("delay", 100.0))
    for k, v in sorted_items:
        sorted_proxy_status[k] = {
            "delay": v.get("delay", 100.0)
        }
        if "is_high_anon" in v:
            sorted_proxy_status[k]["is_high_anon"] = v["is_high_anon"]
        sorted_proxy_status[k]["history_list"] = v.get("history_list", [])

    save_json(PROXY_STATUS_FILE, sorted_proxy_status)

    end_time = time.time()
    print(f"⏱️ 测速与状态更新耗时: {end_time - start_time:.2f} 秒")

    if not valid_proxies_with_time:
        print("❌ 经过 B站 接口测试，没有免费代理存活。")
        return []

    print(
        f"📊 测速完毕，近期存活及可用节点共 {len(valid_proxies_with_time)} 个 (参考存活率: {len(valid_proxies_with_time) / len(unique_proxies) * 100:.2f}%)。")

    # 按平均响应时间升序排序（得分越低越好）
    valid_proxies_with_time.sort(key=lambda x: x[1])

    # 提取前 count 个
    top_proxies_with_time = valid_proxies_with_time[:count] if count else valid_proxies_with_time

    final_proxy_list = []
    print(f"\n🏆 为您提取 Top {len(top_proxies_with_time)} 历史综合最快节点：")
    for idx, (p_dict, avg_score) in enumerate(top_proxies_with_time):
        # print(f"  [{idx + 1}] 平均延迟得分: {avg_score:.3f}秒 ➔ {p_dict['http']}")
        final_proxy_list.append(p_dict)

    return final_proxy_list


# ================= 新增区：高匿复测与状态落地封装模块 =================

def get_real_ip():
    """获取本机真实公网 IP"""
    try:
        resp = requests.get("http://httpbin.org/ip", timeout=10)
        if resp.status_code == 200:
            return resp.json().get("origin", "").split(",")[0].strip()
    except Exception:
        pass
    return None


def _check_single_high_anon(proxy_dict, real_ip):
    """
    内部辅助函数：校验单个代理是否高匿
    返回元组: (proxy_dict, 是否为高匿布尔值)
    """
    try:
        resp = requests.get("http://httpbin.org/ip", proxies=proxy_dict, timeout=5.0)
        if resp.status_code == 200:
            origin_ip = resp.json().get("origin", "")
            if (real_ip and real_ip in origin_ip) or ("," in origin_ip):
                return (proxy_dict, False)
            return (proxy_dict, True)
    except:
        pass
    return (proxy_dict, False)


def batch_update_anon_status(anon_status_records):
    """
    批量将高匿测速结果追加/更新到本地 JSON 文件中
    :param anon_status_records: dict, 格式为 {"1.2.3.4:80": True/False}
    """
    if not anon_status_records:
        return

    proxy_status = read_json(PROXY_STATUS_FILE)
    if not isinstance(proxy_status, dict):
        return

    is_updated = False
    for proxy_str, is_high_anon in anon_status_records.items():
        if proxy_str in proxy_status:
            proxy_status[proxy_str]["is_high_anon"] = is_high_anon
            is_updated = True

    if is_updated:
        # 写入前保持 JSON 数据结构的整洁与有序
        sorted_proxy_status = {}
        sorted_items = sorted(proxy_status.items(), key=lambda x: x[1].get("delay", 100.0))
        for k, v in sorted_items:
            sorted_proxy_status[k] = {
                "delay": v.get("delay", 100.0),
                "is_high_anon": v.get("is_high_anon", False),
                "history_list": v.get("history_list", [])
            }
        save_json(PROXY_STATUS_FILE, sorted_proxy_status)


def filter_and_record_high_anon(best_free_proxies, count):
    """
    核心封装：对传入的精英代理列表进行高匿检测，提取所需数量，并批量持久化结果。
    """
    print("\n🛡️ [安全模式] 正在启动高匿二次过滤并记录状态...")
    real_ip = get_real_ip()
    if real_ip:
        print(f"✅ 获取到本机真实 IP: {real_ip}")
    else:
        print("⚠️ 获取本机真实 IP 失败，将仅使用头部特征(逗号)进行基础匿名过滤。")

    print(f"🔍 正在对 Top {len(best_free_proxies)} 个极速节点进行高匿并发核对...")

    base_proxy_list = []
    anon_status_records = {}  # 收集高匿测速结果，用于最后统一落地

    with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
        future_to_proxy = {
            executor.submit(_check_single_high_anon, p, real_ip): p
            for p in best_free_proxies
        }

        for future in concurrent.futures.as_completed(future_to_proxy):
            p_dict, is_high_anon = future.result()
            # 从 "http://1.2.3.4:80" 中剥离出纯 "1.2.3.4:80" 作为字典 key
            proxy_str = p_dict["http"].replace("http://", "")
            anon_status_records[proxy_str] = is_high_anon

            if is_high_anon and len(base_proxy_list) < count:
                base_proxy_list.append(p_dict)

    # 所有测试跑完后，将这一批节点的匿名状态批量刷入 JSON
    batch_update_anon_status(anon_status_records)

    return base_proxy_list


# =========================================================================

def get_proxy(count=50):
    """
    组装所有可用代理的主函数
    :param count: 最终需要返回的代理数量（分别对应普通和高匿的期望最大获取数）
    :return: 元组 (base_proxy_list, high_anon_proxy_list)
             - base_proxy_list: 仅经过 B 站连通性测速的最快代理列表
             - high_anon_proxy_list: 经过高匿二次校验的最快代理列表
    """
    # 放宽提取容量（取 3 倍），确保经过高匿严格过滤后，依然能凑够 count 数量
    fetch_count = count * 3
    best_free_proxies = get_top_proxies(count=fetch_count)

    if not best_free_proxies:
        return [], []

    # 1. 常规代理列表：不需要高匿，直接截取连通测速最快的前 count 个
    base_proxy_list = best_free_proxies[:count]

    # 2. 高匿代理列表：将扩容后的整个优质池送入独立校验逻辑，检测并记录状态，最多提取 count 个
    high_anon_proxy_list = filter_and_record_high_anon(best_free_proxies, count)

    print(f"\n✅ 最终交付可用代理池容量：常规极速节点 {len(base_proxy_list)} 个，纯高匿节点 {len(high_anon_proxy_list)} 个。")

    return base_proxy_list, high_anon_proxy_list


if __name__ == "__main__":
    # 运行测试 (按需开启高匿验证)
    base_proxy_list, high_anon_proxy_list = get_proxy(count=50)
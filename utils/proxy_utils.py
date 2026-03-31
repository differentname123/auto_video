import requests
import concurrent.futures
import time
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

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

# 【新增配置】数据源最大容忍未更新时间（小时）
# 如果一个源超过 48 小时未更新，大概率变成僵尸源，直接抛弃不拉取
MAX_SOURCE_AGE_HOURS = 48.0

# ==========================================
# 代理源拉取统一配置区
# ==========================================

# 统一维护的代理源配置列表：(来源名称, URL, 数据类型)
PROXY_SOURCES = [
    # 原基础源
    ("ProxyScrape",
     "https://api.proxyscrape.com/v4/free-proxy-list/get?request=displayproxies&protocol=http&anonymity=elite,anonymous",
     "text"),
    ("GitHub (TheSpeedX)", "https://raw.githubusercontent.com/TheSpeedX/PROXY-List/master/http.txt", "text"),
    ("GitHub (monosans)", "https://raw.githubusercontent.com/monosans/proxy-list/main/proxies/http.txt", "text"),
    ("GitHub (proxifly)",
     "https://raw.githubusercontent.com/proxifly/free-proxy-list/main/proxies/protocols/http/data.txt", "text"),

    # 新增的高质量源 (gemini)
    ("GitHub (ShiftyTR)", "https://raw.githubusercontent.com/ShiftyTR/Proxy-List/master/http.txt", "text"),
    ("GitHub (jetkai)", "https://raw.githubusercontent.com/jetkai/proxy-list/main/online-proxies/txt/proxies-http.txt",
     "text"),
    ("spys.me", "https://spys.me/proxy.txt", "text"),

    # grok源
    ("GitHub (clarketm)", "https://raw.githubusercontent.com/clarketm/proxy-list/master/proxy-list-raw.txt", "text"),
    ("GitHub (komutan234)", "https://raw.githubusercontent.com/komutan234/Proxy-List-Free/main/proxies/http.txt",
     "text"),
    ("GitHub (iplocate)", "https://raw.githubusercontent.com/iplocate/free-proxy-list/main/protocols/http.txt", "text"),
    ("GitHub (mmpx12)", "https://raw.githubusercontent.com/mmpx12/proxy-list/master/http.txt", "text"),

    # gpt源
    ("GitHub (zloi-user)", "https://raw.githubusercontent.com/zloi-user/hideip.me/main/http.txt", "text"),
    ("GitHub (prxchk)", "https://raw.githubusercontent.com/prxchk/proxy-list/main/http.txt", "text"),

    # 版本1引入的额外补充源
    ("GitHub (Zaeem20)", "https://raw.githubusercontent.com/Zaeem20/FREE_PROXIES_LIST/master/http.txt", "text"),
    ("GitHub (MuRongPIG)", "https://raw.githubusercontent.com/MuRongPIG/Proxy-Master/main/http.txt", "text"),
    ("GitHub (Anonym0us)",
     "https://raw.githubusercontent.com/Anonym0usWork1221/Free-Proxies/main/proxy_files/http_proxies.txt", "text"),
    ("GitHub (roosterkid)", "https://raw.githubusercontent.com/roosterkid/openproxylist/main/HTTPS_RAW.txt", "text"),
    ("GitHub (vakhov)", "https://raw.githubusercontent.com/vakhov/fresh-proxy-list/master/http.txt", "text"),
    ("GitHub (proxylist-to)", "https://raw.githubusercontent.com/proxylist-to/proxy-list/main/http.txt", "text"),
    ("GitHub (sunny9577)", "https://raw.githubusercontent.com/sunny9577/proxy-scraper/master/proxies.txt", "text"),
    ("API (openproxylist)", "https://api.openproxylist.xyz/http.txt", "text"),
    ("API (proxyspace)", "https://proxyspace.pro/http.txt", "text"),

    # JSON 分页接口格式
    ("Geonode API", "https://proxylist.geonode.com/api/proxy-list", "json_geonode")
]


# ================= 新增区：源鲜度检测模块 =================

def _check_source_freshness(url):
    """
    检查源的最后更新时间，判断是否过期。
    返回: (is_fresh: bool, reason: str)
    """
    try:
        now = datetime.now(timezone.utc)

        # 1. 如果是 GitHub 源，尝试使用 GitHub API 获取最后一次 commit 时间 (最准确)
        github_match = re.match(r'https://raw\.githubusercontent\.com/([^/]+)/([^/]+)/([^/]+)/(.*)', url)
        if github_match:
            owner, repo, branch, path = github_match.groups()
            api_url = f"https://api.github.com/repos/{owner}/{repo}/commits?path={path}&sha={branch}&per_page=1"
            # GitHub API 限制无 Token 为 60次/小时，我们的源数量完全在安全线内
            resp = requests.get(api_url, proxies=LOCAL_PROXY, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if data and isinstance(data, list):
                    commit_date_str = data[0]['commit']['committer']['date']
                    commit_date = datetime.fromisoformat(commit_date_str.replace("Z", "+00:00"))
                    age_hours = (now - commit_date).total_seconds() / 3600
                    if age_hours > MAX_SOURCE_AGE_HOURS:
                        return False, f"过期 {age_hours:.1f}h"
                    return True, f"更新于 {age_hours:.1f}h 前"

        # 2. 如果不是 GitHub，或者 API 失败，退化为使用 HTTP HEAD 检查 Last-Modified
        head_resp = requests.head(url, proxies=LOCAL_PROXY, timeout=5)
        last_modified_str = head_resp.headers.get('Last-Modified')

        if last_modified_str:
            last_modified_time = parsedate_to_datetime(last_modified_str)
            age_hours = (now - last_modified_time).total_seconds() / 3600
            if age_hours > MAX_SOURCE_AGE_HOURS:
                return False, f"过期 {age_hours:.1f}h"
            return True, f"更新于 {age_hours:.1f}h 前"

        # 3. 如果服务器连 Last-Modified 都不给，默认放行，交由后续测速决定
        return True, "未知更新时间"

    except Exception as e:
        # 检测过程报错不应阻塞主流程，给予默认通过
        return True, "检测时间失败"


def _fetch_proxy_from_source(source_name, url, source_type):
    """
    统一代理源拉取引擎：自带多重重试机制，并能兼容绝大多数代理文本格式变种。
    返回值: (source_name, proxies_list, status_message)
    """
    all_proxies = []

    # 先做鲜度检查
    is_fresh, time_msg = _check_source_freshness(url)
    if not is_fresh:
        return source_name, [], f"⚠️ 忽略过期源 ({time_msg})"

    if source_type == "json_geonode":
        # 专门处理 Geonode 的 JSON 分页格式
        for page in range(1, 4):
            page_url = f"{url}?limit=500&page={page}&sort_by=lastChecked&sort_type=desc&protocols=http"
            success = False
            for attempt in range(3):
                try:
                    response = requests.get(page_url, proxies=LOCAL_PROXY, timeout=15)
                    if response.status_code == 200:
                        items = response.json().get("data", [])
                        if not items:
                            success = True
                            break
                        for item in items:
                            ip, port = item.get("ip", ""), item.get("port", "")
                            if ip and port:
                                all_proxies.append(f"{ip}:{port}")
                        success = True
                        break
                except Exception:
                    if attempt < 2: time.sleep(1)
            if not success:
                break
    else:
        # 处理所有 Text 格式 (通用兼容: 附加信息多列、带http前缀、ip:port:country等)
        for attempt in range(5):
            try:
                response = requests.get(url, proxies=LOCAL_PROXY, timeout=15)
                if response.status_code == 200:
                    for line in response.text.split('\n'):
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue

                        # 兼容处理1: 切除空格/逗号后的附加信息 (如 spys.me / proxifly 的第一列格式)
                        entry = line.split()[0].split(',')[0]

                        # 兼容处理2: 强制去除协议头
                        for prefix in ("http://", "https://"):
                            if entry.startswith(prefix):
                                entry = entry[len(prefix):]
                                break

                        # 兼容处理3: 以冒号切分，严格提取前两个组成部分（解决如 ip:port:country 格式）
                        parts = entry.split(':')
                        if len(parts) >= 2 and parts[0][0].isdigit():
                            # 过滤明显的脏数据 (如 0.0.0.0 或者非数字端口)
                            ip, port = parts[0], parts[1]
                            if ip != "0.0.0.0" and port.isdigit():
                                all_proxies.append(f"{ip}:{port}")
                    break  # 成功提取后跳出重试循环
            except Exception:
                if attempt < 4:
                    time.sleep(1)

    if all_proxies:
        msg = f"✅ 成功获取 {len(all_proxies):>5} 个 | 示例: {all_proxies[0]:<21} | 状态: {time_msg}"
    else:
        msg = f"❌ 获取失败或无有效数据 | 状态: {time_msg}"

    return source_name, all_proxies, msg


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


def get_top_proxies(count=10):
    """
    主控函数：多源拉取 -> 去重 -> 多线程测速 -> 记录历史得分 -> 排序 -> 提取 Top N
    """
    print("=== 第一阶段：从各个数据源聚合代理 ===")

    raw_proxies = []
    print(f"🚀 启动并发拉取，共计 {len(PROXY_SOURCES)} 个数据源...")

    # 启用线程池并发拉取所有数据源，大大降低总体等待时间
    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
        futures = [executor.submit(_fetch_proxy_from_source, name, url, stype) for name, url, stype in PROXY_SOURCES]

        for future in concurrent.futures.as_completed(futures):
            source_name, proxy_list, status_msg = future.result()
            print(f"  └─ {status_msg} | 来源: {source_name}")
            if proxy_list:
                raw_proxies.extend(proxy_list)

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

    print(
        f"\n✅ 最终交付可用代理池容量：常规极速节点 {len(base_proxy_list)} 个，纯高匿节点 {len(high_anon_proxy_list)} 个。")

    return base_proxy_list, high_anon_proxy_list


if __name__ == "__main__":
    # 运行测试 (按需开启高匿验证)
    base_proxy_list, high_anon_proxy_list = get_proxy(count=50)
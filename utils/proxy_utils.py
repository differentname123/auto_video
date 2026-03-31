import requests
import concurrent.futures
import time

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
                # 如果是纯 ip:port 则直接用，如果有附加信息则提取第一段
                proxies = [line.split()[0].split(',')[0] for line in lines]
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
    测试并返回：(代理字典, 响应时间)
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
            return (proxy_dict, elapsed_time)
    except:
        pass
    return None


def get_top_proxies(count=10):
    """
    主控函数：多源拉取 -> 去重 -> 多线程测速 -> 排序 -> 提取 Top N
    """
    print("=== 第一阶段：从各个数据源聚合代理 ===")

    # 【灵活启用/关闭模块】：将不想用的函数注释掉即可
    sources = [
        fetch_from_proxyscrape,
        fetch_from_github_thespeedx,
        fetch_from_github_monosans,
        fetch_from_github_proxifly
    ]

    raw_proxies = []
    for source_func in sources:
        raw_proxies.extend(source_func())

    # 列表去重（不同源之间可能有大量重复的公共免费IP）
    unique_proxies = list(set(raw_proxies))
    print(f"\n🔄 聚合完毕，去重后共获得 {len(unique_proxies)} 个独立代理 IP。")

    if not unique_proxies:
        print("❌ 没有任何原始代理，退出。")
        return []

    print(f"\n=== 第二阶段：并发测速 (宽容超时: {BASE_TIMEOUT}秒) ===")
    valid_proxies_with_time = []

    # 由于 GitHub 源加起来可能有上万个 IP，建议把最大线程数拉高到 300-500 以加快速度
    max_threads = min(500, len(unique_proxies))
    print(f"🚀 启动 {max_threads} 个线程进行轰炸式测速...")

    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
        results = executor.map(format_and_test_proxy, unique_proxies)
        for res in results:
            if res is not None:
                valid_proxies_with_time.append(res)

    end_time = time.time()
    print(f"⏱️ 测速耗时: {end_time - start_time:.2f} 秒")

    if not valid_proxies_with_time:
        print("❌ 经过 B站 接口测试，没有免费代理存活。")
        return []

    print(
        f"📊 测速完毕，仅有 {len(valid_proxies_with_time)} 个节点存活 (存活率: {len(valid_proxies_with_time) / len(unique_proxies) * 100:.2f}%)。")

    # 按响应时间升序排序
    valid_proxies_with_time.sort(key=lambda x: x[1])

    # 提取前 count 个
    top_proxies_with_time = valid_proxies_with_time[:count] if count else valid_proxies_with_time

    final_proxy_list = []
    print(f"\n🏆 为您提取 Top {len(top_proxies_with_time)} 最快节点：")
    for idx, (p_dict, t_time) in enumerate(top_proxies_with_time):
        # print(f"  [{idx + 1}] 延迟: {t_time:.3f}秒 ➔ {p_dict['http']}")
        final_proxy_list.append(p_dict)

    return final_proxy_list


def get_proxy(count=50):
    """组装所有可用代理的主函数"""
    base_proxy_list = []

    # 动态抓取并测试最新的免费代理 (建议每次只要前10-20个最快的，因为免费代理死得快)
    best_free_proxies = get_top_proxies(count=count)

    if best_free_proxies:
        base_proxy_list.extend(best_free_proxies)

    print(f"\n✅ 最终可用代理池容量：{len(base_proxy_list)} 个。")
    return base_proxy_list


if __name__ == "__main__":
    # 运行测试
    proxy_list = get_proxy()
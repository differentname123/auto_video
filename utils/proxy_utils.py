import requests

import concurrent.futures

# ================= 配置区 =================
# 您的本地翻墙代理（仅用于访问 ProxyScrape 接口一次）
LOCAL_PROXY = {
    "http": "http://127.0.0.1:7890",
    "https": "http://127.0.0.1:7890"
}

# 测试代理存活的目标网站（B站基础接口）
TEST_TARGET_URL = "https://api.bilibili.com/x/web-interface/nav"

# 基础超时宽容度（秒）：超过这个时间的代理直接判定死亡，不再参与排序
BASE_TIMEOUT = 15.0


# ==========================================

def fetch_raw_proxies():
    """
    通过本地代理仅获取一次原始列表
    """
    print("正在通过本地代理 (127.0.0.1:7890) 拉取原始数据...")
    url = "https://api.proxyscrape.com/v4/free-proxy-list/get"
    params = {
        "request": "displayproxies",
        "protocol": "http",
        "anonymity": "elite,anonymous"
    }

    try:
        response = requests.get(url, params=params, proxies=LOCAL_PROXY, timeout=15)
        if response.status_code == 200:
            raw_list = [p.strip() for p in response.text.split('\n') if p.strip()]
            print(f"✅ 成功拉取到 {len(raw_list)} 个原始代理。")
            return raw_list
    except Exception as e:
        print(f"❌ 原始列表拉取失败: {e}")
    return []


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
        # 发起请求，并在设定的宽容超时时间内等待
        response = requests.get(
            TEST_TARGET_URL,
            proxies=proxy_dict,
            headers=headers,
            timeout=BASE_TIMEOUT
        )

        if response.status_code == 200 and "code" in response.json():
            # 获取该次请求的真实耗时（秒）
            elapsed_time = response.elapsed.total_seconds()
            return (proxy_dict, elapsed_time)

    except:
        pass

    return None


def get_top_proxies(count=None):
    """
    主控函数：拉取 -> 多线程测速 -> 排序 -> 提取 Top N
    :param count: 需要返回的代理数量。如果不传，则返回所有存活的节点。
    """
    raw_proxies = fetch_raw_proxies()
    if not raw_proxies:
        return []

    print(f"\n🚀 开始并发测速 (宽容超时: {BASE_TIMEOUT}秒)...")

    valid_proxies_with_time = []

    # 开启 100 线程测速
    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
        results = executor.map(format_and_test_proxy, raw_proxies)

        for res in results:
            if res is not None:
                valid_proxies_with_time.append(res)

    # 如果一个都没活下来，直接返回空
    if not valid_proxies_with_time:
        print("❌ 没有代理存活。")
        return []

    print(f"\n📊 测速完毕，共有 {len(valid_proxies_with_time)} 个节点存活。正在按延迟排序...")

    # 【核心逻辑】：按照元组的第二个元素（即 elapsed_time 响应时间）进行升序排序 (越快越靠前)
    valid_proxies_with_time.sort(key=lambda x: x[1])

    # 提取前 count 个（如果 count 为空或者大于存活总数，则全取）
    if count is not None:
        top_proxies_with_time = valid_proxies_with_time[:count]
    else:
        top_proxies_with_time = valid_proxies_with_time

    # 剥离时间数据，只返回您需要的代理字典列表
    final_proxy_list = []
    print(f"\n🏆 为您提取 Top {len(top_proxies_with_time)} 最快节点：")
    for idx, (p_dict, t_time) in enumerate(top_proxies_with_time):
        # print(f"  [{idx + 1}] 耗时: {t_time:.3f}秒 ➔ {p_dict['http']}")
        final_proxy_list.append(p_dict)

    return final_proxy_list


def get_proxy():
    """"""
    base_proxy_list = [
        {"http": "http://115.190.54.74:8888", "https": "http://115.190.54.74:8888"},

        {"http": "http://viyvlyeo:lfklf4e2v9qm@31.59.20.176:6754",
         "https": "http://viyvlyeo:lfklf4e2v9qm@31.59.20.176:6754"},
        {"http": "http://viyvlyeo:lfklf4e2v9qm@23.95.150.145:6114",
         "https": "http://viyvlyeo:lfklf4e2v9qm@23.95.150.145:6114"},
        {"http": "http://viyvlyeo:lfklf4e2v9qm@198.23.239.134:6540",
         "https": "http://viyvlyeo:lfklf4e2v9qm@198.23.239.134:6540"},
        {"http": "http://viyvlyeo:lfklf4e2v9qm@45.38.107.97:6014",
         "https": "http://viyvlyeo:lfklf4e2v9qm@45.38.107.97:6014"},
        {"http": "http://viyvlyeo:lfklf4e2v9qm@107.172.163.27:6543",
         "https": "http://viyvlyeo:lfklf4e2v9qm@107.172.163.27:6543"},
        {"http": "http://viyvlyeo:lfklf4e2v9qm@198.105.121.200:6462",
         "https": "http://viyvlyeo:lfklf4e2v9qm@198.105.121.200:6462"},
        {"http": "http://viyvlyeo:lfklf4e2v9qm@64.137.96.74:6641",
         "https": "http://viyvlyeo:lfklf4e2v9qm@64.137.96.74:6641"},
        {"http": "http://viyvlyeo:lfklf4e2v9qm@216.10.27.159:6837",
         "https": "http://viyvlyeo:lfklf4e2v9qm@216.10.27.159:6837"},
        {"http": "http://viyvlyeo:lfklf4e2v9qm@142.111.67.146:5611",
         "https": "http://viyvlyeo:lfklf4e2v9qm@142.111.67.146:5611"},
        {"http": "http://viyvlyeo:lfklf4e2v9qm@191.96.254.138:6185",
         "https": "http://viyvlyeo:lfklf4e2v9qm@191.96.254.138:6185"}
    ]  # 来源于https://dashboard.webshare.io/
    best_proxies = get_top_proxies()  # 来源于 https://docs.proxyscrape.com/api-reference/public-api/get-proxy-list
    if best_proxies:
        # 加入到base_proxy_list中
        base_proxy_list.extend(best_proxies)

    print(f"获取到 {len(base_proxy_list)} 个代理（包括预设和测速后的）。")
    return base_proxy_list


if __name__ == "__main__":
    # 调用时传入 count 参数，例如获取最快的 5 个代理
    # 注意：确保不要在当前脚本中使用 os.environ 全局代理！
    best_proxies = get_top_proxies(count=5)

    print("-" * 50)
    print("最终返回的规范化代理列表：")
    print(best_proxies)

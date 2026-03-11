import json
import os
import time
import urllib.parse
from playwright.sync_api import sync_playwright, Page, expect

# ==============================================================================
# 配置区域
# ==============================================================================
USER_DATA_DIR = r"W:\temp\alimama_data"  # 保存淘宝/阿里妈妈登录状态的目录
TARGET_NUM = 30  # 想要抓取的商品总数


def login_and_save_session():
    """
    首次运行：启动浏览器进行手动登录阿里妈妈，保存登录态。
    """
    if not os.path.exists(USER_DATA_DIR):
        os.makedirs(USER_DATA_DIR)

    print(f"--- 启动浏览器进行手动登录 (数据保存在 {USER_DATA_DIR}) ---")
    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            channel="chrome",  # 【关键1】强制调用本地安装的 Google Chrome 正式版
            user_data_dir=USER_DATA_DIR,  # <-- 使用传入的参数
            headless=False,  # 调试时建议开启 False，稳定后可改为 True
            args=['--disable-blink-features=AutomationControlled', '--start-maximized', '--disable-gpu',
                  '--window-position=0,0'],
            ignore_default_args=["--enable-automation"]
        )
        page = context.pages[0] if context.pages else context.new_page()
        page.goto("https://pub.alimama.com/portal/v2/pages/promo/goods/index.htm")

        print("\n" + "=" * 60)
        print("浏览器已打开，请手动登录阿里妈妈（建议使用扫码登录）。")
        print("登录成功并看到控制台主页后，回到本窗口按 Enter 键继续...")
        print("=" * 60)
        input()
        context.close()
        print("[+] 登录态已保存！可以开始抓取数据了。")


def check_and_handle_quick_entry(page: Page) -> bool:
    """
    【终极主动出击函数 - 跨页面连击版】
    完美应对点击后跳转新页面，新页面仍有“快速进入”的套娃情况。
    """
    clicked = False
    max_attempts = 5
    attempts = 0
    # print("【🔍 主动出击】正在检测是否存在 '快速进入' 挡板...")
    # 定位器：动态绑定，不受页面刷新影响
    btn = page.locator("button:has-text('快速进入')")

    try:
        # 第一层判断：瞬间检测。如果没有，直接 0 延迟返回，完全不影响正常的页面抓取速度
        if not btn.is_visible():
            return False

        while attempts < max_attempts:
            if btn.is_visible():
                print(f"【🚀 主动出击】发现'确认登录'页面 (第 {attempts + 1} 层)，立刻点击 '快速进入'...")
                btn.click()
                clicked = True
                attempts += 1

                # 【🌟 核心突破点】：点击后页面会刷新或跳转，DOM 会短暂消失（白屏）。
                # 绝对不能用 time.sleep()，因为你不知道新页面几秒能出来。
                # 使用 wait_for，让 Playwright 主动在接下来的 3 秒内盯着这个按钮看它会不会"复活"。
                try:
                    btn.wait_for(state="visible", timeout=3000)
                except:
                    # 如果 3 秒内这个按钮没有再出现，说明我们已经跳出了验证死循环，进入了正常页面
                    print("【✅ 破盾成功】'快速进入'按钮已彻底消失。")
                    break
            else:
                break

        if attempts >= max_attempts:
            print("【⚠️ 警告】点击次数达到上限 (5次)！账号登录状态可能已彻底失效，建议重新扫码。")

    except Exception as e:
        print(f"[-] 检测'快速进入'时发生微小异常 (通常是页面刷新瞬间的正常报错，可忽略): {e}")

    return clicked


class AlimamaScraper:
    """
    阿里妈妈数据抓取类，支持保持浏览器窗口复用
    """

    def __init__(self, user_data_dir: str = USER_DATA_DIR):
        if not os.path.exists(user_data_dir):
            raise FileNotFoundError(
                f"[-] 找不到用户数据目录 {user_data_dir}，请先运行 login_and_save_session() 进行登录！")

        print("--- 初始化浏览器实例，准备复用窗口 ---")
        # 手动启动 playwright，而不是使用 with 上下文，确保对象存活时窗口不关闭
        self.playwright = sync_playwright().start()
        self.context = self.playwright.chromium.launch_persistent_context(
            channel="chrome",
            user_data_dir=user_data_dir,
            headless=False,
            args=['--disable-blink-features=AutomationControlled', '--start-maximized', '--disable-gpu',
                  '--window-position=0,0'],
            ignore_default_args=["--enable-automation"]
        )

        self.page = self.context.pages[0] if self.context.pages else self.context.new_page()
        self.page.set_default_timeout(30000)

        # 🚀 主动出击：初始化时先跳转到阿里妈妈首页，排查并点掉可能存在的全局阻拦弹窗
        print("--- 执行初始化环境检查 ---")
        self.page.goto("https://pub.alimama.com/portal/v2/pages/promo/goods/index.htm")
        check_and_handle_quick_entry(self.page)
        print("--- 浏览器初始化完毕，随时可执行搜索 ---")

    def fetch_data(self, search_query: str, target_num: int = 10) -> list:
        all_products = []
        # print(f"\n--- 开始抓取关键词: '{search_query}', 目标数量: {target_num} ---")
        current_page = 1

        while len(all_products) < target_num:
            # print(f"[*] 正在尝试获取第 {current_page} 页的数据...")
            safe_query = urllib.parse.quote(search_query)
            target_url = f"https://pub.alimama.com/portal/v2/pages/promo/goods/index.htm?pageNum={current_page}&pageSize=60&fn=search&q={safe_query}&sort=default"

            result_json = None

            try:
                with self.page.expect_response(
                        lambda response: "skyleap.distribution.site.json" in response.url and response.request.method == "POST",
                        timeout=15000
                ) as response_info:

                    self.page.goto(target_url)

                    # ==========================================================
                    # 🚀 主动出击点 1：页面刚 load 完，立刻检查是不是保活弹窗！
                    # 如果是，点掉它。点击动作会触发新的网络请求，外层的 expect_response 依然能捕获到。
                    # ==========================================================
                    check_and_handle_quick_entry(self.page)

                # 走到这里说明 15 秒内一定拿到了想要的接口数据
                response = response_info.value
                if response.status == 200:
                    result_json = response.json()

            except Exception as e:
                # ==========================================================
                # 保留的兜底逻辑 (被动防御)：如果15秒后还是超时了，再做全面排查
                # ==========================================================

                # 兜底排查 1：万一刚才没点到（网络卡顿），再查一次
                if check_and_handle_quick_entry(self.page):
                    continue  # 重新回到循环起点发起 goto

                # 兜底排查 2：滑块风控
                elif self.page.locator(
                        "#baxia-dialog-content").is_visible() or "sec.taobao.com" in self.page.url or self.page.locator(
                        ".nc_wrapper").is_visible():
                    print("【⚠️ 警告】检测到淘宝风控滑块！请在浏览器界面中手动滑动...")
                    print("系统将等待 20 秒让你完成操作...")
                    self.page.wait_for_timeout(20000)
                    continue

                else:
                    print(f"[-] 15秒内未拉取到数据且未发现已知拦截。原因: {e}")
                    break

            # 阿里妈妈底层 API 返回值的风控判断
            if result_json and 'ret' in result_json and 'FAIL_SYS_USER_VALIDATE' in str(result_json['ret']):
                print("【⚠️ 接口被拦截】触发底层风控滑块！请在浏览器界面中手动滑动。")
                self.page.wait_for_timeout(20000)
                continue

            # --- 下面是数据解析逻辑 ---
            if not result_json:
                continue

            site_data = result_json.get('data', {}).get('siteData', {})
            resultList = site_data.get('resultList', []) if isinstance(site_data, dict) else []

            if not resultList:
                print("[!] API返回数据为空，可能已达最后一页或无商品，停止翻页。")
                break

            extracted_list = []
            for item in resultList:
                pict_url = item.get("pic", "")
                if pict_url and pict_url.startswith("//"):
                    pict_url = "https:" + pict_url

                click_url = item.get("udf_temp_store", {}).get("clickUrl", "")
                if not click_url:
                    click_url = item.get("url", "")
                if click_url and click_url.startswith("//"):
                    click_url = "https:" + click_url

                extracted_item = {
                    "item_id": item.get("outputMktId", ""),
                    "item_name": item.get("itemName", ""),
                    "brand": item.get("brandName", ""),
                    "pict_url": pict_url,
                    "category_name": item.get("categoryName", ""),
                    "level_one_category_name": item.get("levelOneCategoryName", ""),
                    "shop_title": item.get("shopTitle", ""),
                    "short_title": item.get("shortTitle", ""),
                    "original_price": item.get("reservePrice", ""),
                    "final_price": item.get("unionPromotionPrice",
                                            item.get("promotionPrice", item.get("zkFinalPrice", ""))),
                    "commission_rate": item.get("tkRate", ""),
                    "commission_amount": item.get("unionCommissionAmount", item.get("tkCommissionAmount", "")),
                    "click_url": click_url,
                }
                extracted_list.append(extracted_item)

            all_products.extend(extracted_list)
            # print(f"[+] 成功获取 {len(extracted_list)} 条商品。当前总数: {len(all_products)} / {target_num}")

            current_page += 1
            time.sleep(2)

        final_results = all_products
        # print(f"✅ 抓取完成！本关键词共返回 {len(final_results)} 条商品。")
        return final_results

    def close(self):
        """抓取完毕后，安全关闭并清理资源"""
        print("\n--- 任务全部结束，正在清理并关闭浏览器 ---")
        self.context.close()
        self.playwright.stop()


if __name__ == '__main__':
    # # 步骤 1：第一次使用时，请取消下面这行的注释，运行以登录阿里妈妈并保存状态
    # login_and_save_session()

    # 步骤 2：登录完成后，注释掉上面那行，运行拉取代码

    # 【改动体现】初始化浏览器类，浏览器在此刻打开
    scraper = AlimamaScraper()

    try:
        # 同一个对象，同一个窗口，执行多次搜索请求
        results_1 = scraper.fetch_data("可乐", target_num=10)
        results_2 = scraper.fetch_data("香蕉", target_num=10)
        results_3 = scraper.fetch_data("苹果", target_num=10)

        # 打印部分结果检查一下
        print("\n【可乐搜索结果前2条】:")
        for res in results_1[:2]:
            print(json.dumps(res, ensure_ascii=False, indent=2))

    finally:
        # 无论抓取是否发生异常，最后都安全关闭浏览器进程防内存泄漏
        scraper.close()
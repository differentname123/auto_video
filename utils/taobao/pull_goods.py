import time
import hashlib
import json
from urllib import parse, request


def generate_sign(secret, params):
    """生成淘宝 API 签名 (MD5 算法)"""
    # 1. 参数名按字典序排序
    sorted_params = sorted(params.items())

    # 2. 拼接字符串：secret + key1+value1 + key2+value2 + ... + secret
    query_str = secret
    for k, v in sorted_params:
        query_str += f"{k}{v}"
    query_str += secret

    # 3. 计算 MD5 并转大写
    m = hashlib.md5()
    m.update(query_str.encode('utf-8'))
    return m.hexdigest().upper()


def get_tbk_material(app_key, app_secret, adzone_id, q=None, material_id=80309, desire_count=50, **kwargs):
    """获取淘宝客商品物料，脱离官方 SDK 手动实现"""
    url = "http://gw.api.taobao.com/router/rest"

    all_map_data = []
    page_no = 1

    # 清理 kwargs 中可能误传的分页参数，交由内部自动接管
    kwargs.pop('page_no', None)
    kwargs.pop('page_size', None)

    while len(all_map_data) < desire_count:
        # 计算当前这页需要拉取多少条数据 (淘宝 API 限制 page_size 范围是 1~100)
        remaining_count = desire_count - len(all_map_data)
        current_page_size = min(100, remaining_count)

        # 构造公共参数和业务参数
        params = {
            "method": "taobao.tbk.dg.material.optional.upgrade",
            "app_key": app_key,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "format": "json",
            "v": "2.0",
            "sign_method": "md5",
            "adzone_id": str(adzone_id),
            "material_id": str(material_id),
            "page_no": str(page_no),
            "page_size": str(current_page_size)
        }

        if q:
            params["q"] = q

        # 处理额外传入的参数 (转换布尔值为字符串，保证签名正确)
        for k, v in kwargs.items():
            if isinstance(v, bool):
                params[k] = "true" if v else "false"
            else:
                params[k] = str(v)

        # 计算并添加签名
        params["sign"] = generate_sign(app_secret, params)

        # 发送 POST 请求
        try:
            data = parse.urlencode(params).encode('utf-8')
            req = request.Request(url, data=data)
            with request.urlopen(req) as response:
                res_data = response.read().decode('utf-8')
                res_json = json.loads(res_data)

                # 拦截并打印业务级别 API 错误
                if "error_response" in res_json:
                    err = res_json["error_response"]
                    print(
                        f"[API业务错误-物料搜索] 停止拉取 | 错误码: {err.get('code')} | 描述: {err.get('msg')} | 子错误码: {err.get('sub_code')} | 子描述: {err.get('sub_msg')}")
                    break

                # 逐层安全提取目标数据 map_data
                resp_node = res_json.get("tbk_dg_material_optional_upgrade_response", {})
                result_list = resp_node.get("result_list") or {}

                if isinstance(result_list, dict):
                    map_data = result_list.get("map_data", [])
                else:
                    map_data = []

                # 如果当前页已经没有数据了，直接结束循环
                if not map_data:
                    break

                # ==== 核心提取逻辑：结构化你最关心的字段 ====
                for item in map_data:
                    # 安全获取几个核心字典节点，防止 None 导致崩溃
                    price_info = item.get("price_promotion_info", {})
                    basic_info = item.get("item_basic_info", {})
                    publish_info = item.get("publish_info", {})
                    income_info = publish_info.get("income_info", {})

                    # 组装扁平化数据
                    extracted_item = {
                        "item_name": basic_info.get("title", ""),
                        "brand": basic_info.get("brand_name", ""),
                        "original_price": price_info.get("reserve_price", ""),
                        "final_price": price_info.get("final_promotion_price", price_info.get("zk_final_price", "")),
                        "commission_rate": income_info.get("commission_rate", ""),
                        "commission_amount": income_info.get("commission_amount", ""),
                        # 为了第二步能够顺利转链，这里把推广链接顺手提取出来 (优先取二合一券链接，没有则取普通链接)
                        "click_url": publish_info.get("coupon_share_url", publish_info.get("click_url", "")),
                        "raw_data": item
                    }
                    all_map_data.append(extracted_item)

                page_no += 1

        except json.JSONDecodeError as e:
            print(f"[JSON解析异常-物料搜索] 停止拉取 | 详情: {e} | 原始返回: {res_data}")
            break
        except Exception as e:
            print(f"[网络或系统异常-物料搜索] 停止拉取 | 详情: {e}")
            break

    # 截取精确的数量返回
    return all_map_data[:desire_count]


def create_tbk_tpwd(app_key, app_secret, target_url):
    """根据目标链接生成淘口令"""
    url = "http://gw.api.taobao.com/router/rest"

    # 淘宝返回的链接常常是 // 开头的，生成淘口令要求必须是 https 开头
    if target_url and target_url.startswith("//"):
        target_url = "https:" + target_url

    params = {
        "method": "taobao.tbk.tpwd.create",
        "app_key": app_key,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "format": "json",
        "v": "2.0",
        "sign_method": "md5",
        "url": target_url
    }

    params["sign"] = generate_sign(app_secret, params)

    try:
        data = parse.urlencode(params).encode('utf-8')
        req = request.Request(url, data=data)
        with request.urlopen(req) as response:
            res_data = response.read().decode('utf-8')
            res_json = json.loads(res_data)

            # 错误拦截
            if "error_response" in res_json:
                err = res_json["error_response"]
                print(
                    f"[API业务错误-淘口令生成] 失败 | 错误码: {err.get('code')} | 描述: {err.get('msg')} | 子错误码: {err.get('sub_code')} | 子描述: {err.get('sub_msg')}")
                return None

            # 提取数据节点
            resp_node = res_json.get("tbk_tpwd_create_response", {})
            return resp_node.get("data", {})

    except Exception as e:
        print(f"[网络或系统异常-淘口令生成] 详情: {e}")
        return None


def batch_append_tpwd(app_key, app_secret, item_list):
    """批量为商品列表追加淘口令信息"""
    for item in item_list:
        link = item.get("click_url")
        if link:
            tpwd_result = create_tbk_tpwd(app_key, app_secret, link)
            if tpwd_result:
                item["tpwd_simple"] = tpwd_result.get("password_simple", "")
                item["tpwd_model"] = tpwd_result.get("model", "")
            else:
                item["tpwd_simple"] = "生成失败"
                item["tpwd_model"] = "生成失败"
        else:
            item["tpwd_simple"] = "无推广链接"
            item["tpwd_model"] = "无推广链接"

    return item_list


# ================= 使用示例 =================
if __name__ == "__main__":
    APP_KEY = "35288611"
    APP_SECRET = "3f861aab50cd92d29870becf34d6fc64"
    ADZONE_ID = 116238150030

    # ============ 第一步：先搜索商品 ============
    result_list = get_tbk_material(
        app_key=APP_KEY,
        app_secret=APP_SECRET,
        adzone_id=ADZONE_ID,
        q="女装",
        desire_count=3  # 测试拉取 3 个商品看下格式
    )

    print(f"成功获取到了 {len(result_list)} 个商品。")

    # ============ 第二步：为获取到的商品追加生成淘口令 ============
    result_list = batch_append_tpwd(APP_KEY, APP_SECRET, result_list)
    print("追加淘口令后的结果示例：")

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
                        f"[API业务错误] 停止拉取 | 错误码: {err.get('code')} | 描述: {err.get('msg')} | 子错误码: {err.get('sub_code')} | 子描述: {err.get('sub_msg')}")
                    break

                # 逐层安全提取目标数据 map_data
                resp_node = res_json.get("tbk_dg_material_optional_upgrade_response", {})
                result_list = resp_node.get("result_list") or {}

                # 淘宝客接口有时没数据会返回空字符串而非字典，此处需要判断防崩溃
                if isinstance(result_list, dict):
                    map_data = result_list.get("map_data", [])
                else:
                    map_data = []

                # 如果当前页已经没有数据了，直接结束循环
                if not map_data:
                    break

                all_map_data.extend(map_data)
                page_no += 1

        except json.JSONDecodeError as e:
            print(f"[JSON解析异常] 停止拉取 | 详情: {e} | 原始返回: {res_data}")
            break
        except Exception as e:
            print(f"[网络或系统异常] 停止拉取 | 详情: {e}")
            break

    # 截取精确的数量返回（防止最后一次请求拉取超量）
    return all_map_data[:desire_count]


# ================= 使用示例 =================
if __name__ == "__main__":
    APP_KEY = "35288611"
    APP_SECRET = "3f861aab50cd92d29870becf34d6fc64"
    ADZONE_ID = 116238150030  # 替换成你真实的推广位数字

    # 现在只需传入 desire_count，底层会自动计算页数并组装数据
    result_list = get_tbk_material(
        app_key=APP_KEY,
        app_secret=APP_SECRET,
        adzone_id=ADZONE_ID,
        q="可口可乐",
        desire_count=150  # 尝试拉取 150 个商品
    )

    print(f"成功获取到了 {len(result_list)} 个商品。")
    if result_list:
        # 仅打印第一个商品作为验证
        print("第一个商品示例信息:")
        print(json.dumps(result_list[0], indent=4, ensure_ascii=False))
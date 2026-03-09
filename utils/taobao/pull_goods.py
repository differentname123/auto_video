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


def get_tbk_material(app_key, app_secret, adzone_id, q=None, material_id=80309, **kwargs):
    """获取淘宝客商品物料，脱离官方 SDK 手动实现"""
    url = "http://gw.api.taobao.com/router/rest"

    # 构造公共参数和业务参数
    params = {
        "method": "taobao.tbk.dg.material.optional.upgrade",
        "app_key": app_key,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "format": "json",
        "v": "2.0",
        "sign_method": "md5",
        "adzone_id": str(adzone_id),
        "material_id": str(material_id)
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
            return json.loads(res_data)
    except Exception as e:
        print(f"请求失败: {e}")
        return None


# ================= 使用示例 =================
if __name__ == "__main__":
    APP_KEY = "35288611"
    APP_SECRET = "3f861aab50cd92d29870becf34d6fc64"
    ADZONE_ID = 116238150030  # 替换成你真实的推广位数字

    result = get_tbk_material(
        app_key=APP_KEY,
        app_secret=APP_SECRET,
        adzone_id=ADZONE_ID,
        q="女装",
        page_no=1,
        page_size=50
    )

    # 打印格式化后的结果
    print(json.dumps(result, indent=4, ensure_ascii=False))
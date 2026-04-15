import hashlib
import json
import random
import traceback
from datetime import datetime

import requests
import base64
import os
import math
import time
import urllib.parse

from utils.bilibili.get_danmu import get_cid_from_bvid
from utils.common_utils import get_config, init_config, read_json, save_json


# --- 新增的并发控制锁 ---
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


# --- 配置参数 ---
SESSDATA = get_config("xiaoxiaosu_bilibili_sessdata_cookie")  # 必需。你的B站登录会话 SESSDATA cookie 值。
BILI_JCT = get_config("xiaoxiaosu_bilibili_csrf_token")
full_cookie = get_config("xiaoxiaosu_bilibili_total_cookie")
# 默认投稿设置
DEFAULT_COPYRIGHT = 1  # 1: 自制, 2: 转载
DEFAULT_TID = 21  # 21-日常
DEFAULT_RECREATE = -1  # -1: 允许二创, 1: 不允许
DEFAULT_DYNAMIC = "我的第一个B站投稿，希望大家喜欢！"
DEFAULT_NO_REPRINT = 1  # 1: 允许转载, 0: 不允许

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Connection": "keep-alive",
}

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36 Edg/139.0.0.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:142.0) Gecko/20100101 Firefox/142.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36 QuarkPC/4.4.5.505",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36"
]


def get_deterministic_ua(bili_jct: str) -> str:
    """
    根据 bili_jct 计算一个确定的 User-Agent。
    """
    # 1. 使用 hashlib.md5 对 bili_jct 进行哈希。MD5对于这种非安全场景足够了，且速度快。
    #    必须先将字符串编码为字节串。
    hasher = hashlib.md5(bili_jct.encode('utf-8'))

    # 2. 获取哈希值的十六进制表示，并将其转换为一个大整数。
    hash_as_int = int(hasher.hexdigest(), 16)

    # 3. 使用这个大整数对 User-Agent 列表的长度取模，得到一个稳定的索引。
    index = hash_as_int % len(USER_AGENTS)

    return USER_AGENTS[index]


def get_session(sessdata, bili_jct, full_cookie) -> requests.Session:
    sess = requests.Session()

    session_headers = HEADERS.copy()
    # 调用新函数来获取确定的UA
    session_headers['User-Agent'] = get_deterministic_ua(bili_jct)
    session_headers['Cookie'] = full_cookie

    sess.headers.update(session_headers)
    sess.cookies.set("SESSDATA", sessdata)
    sess.cookies.set("bili_jct", bili_jct)
    return sess


def upload_cover(session: requests.Session, image_path: str, bili_jct=BILI_JCT) -> str:
    """
    上传封面图片，返回封面 URL。
    """
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode()

    cover_base64 = f"data:image/jpeg;base64,{data}"
    resp = session.post(
        "https://member.bilibili.com/x/vu/web/cover/up",
        params={"ts": int(time.time() * 1000)},
        data={"csrf": bili_jct, "cover": cover_base64}
    )
    resp.raise_for_status()
    result = resp.json()
    if result.get("code") != 0:
        raise RuntimeError(f"封面上传失败：{result.get('message')}")
    return result["data"]["url"]


def fetch_bili_topics(cookie: str, type_pid=1029, type_id=21):
    """
    使用给定的 Cookie 获取 B 站创作中心话题类型信息。

    参数:
        cookie (str): 用于请求的完整 Cookie。

    返回:
        dict: 返回接口返回的 JSON 数据，如果请求失败则返回 None。
    """
    base_url = "https://member.bilibili.com/x/vupre/web/topic/type/v2"

    # 请求参数
    params = {
        "pn": 0,
        "ps": 200,
        "platform": "pc",
        "type_id": 21,
        "type_pid": type_pid,
    }

    # 构造请求头
    headers = {
        "accept": "application/json, text/plain, */*",
        "accept-language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
        "sec-ch-ua": "\"Microsoft Edge\";v=\"137\", \"Chromium\";v=\"137\", \"Not/A)Brand\";v=\"24\"",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "\"Windows\"",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "Referer": "https://member.bilibili.com/platform/upload/video/frame?page_from=creative_home_top_upload",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36 Edg/137.0.0.0",
        "Cookie": cookie
    }

    try:
        session = requests.Session()
        response = session.get(base_url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
        if 'response' in locals() and response is not None:
            print(f"状态码: {response.status_code}")
            print(f"响应文本: {response.text}")
        return None


def preupload_video(session: requests.Session, video_path: str) -> dict:
    """
    预上传视频，获取上传参数。
    """
    size = os.path.getsize(video_path)
    name = os.path.basename(video_path)

    user_conf = read_json(r'W:\project\python_project\auto_video\config\user_config.json')
    upcdn_list = user_conf.get('upcdn_list', ["estx", "akbd"])

    # 独占模式获取最优节点
    upcdn = get_best_upcdn(upcdn_list, size)

    params = {
        "probe_version": "20250923",
        "upcdn": upcdn,
        "zone": "cs",
        "name": name,
        "r": "upos",
        "profile": "ugcfx/bup",
        "ssl": "0",
        "version": "2.14.0.0",
        "build": "2140000",
        "size": size,
        "webVersion": "2.14.0",
    }

    headers = {
        "Referer": "https://member.bilibili.com/platform/upload/video/frame"
    }

    resp = session.get(
        "https://member.bilibili.com/preupload",
        params=params,
        headers=headers,
        timeout=30
    )

    resp.raise_for_status()
    data = resp.json()
    if data.get("OK") != 1:
        raise RuntimeError(f"预上传失败：{data}")

    # 悄悄携带选中的 upcdn 给主流程，用于记录耗时
    data['_chosen_upcdn'] = upcdn

    return data


def post_video_meta(session: requests.Session, pre: dict, video_path: str) -> dict:
    """
    提交视频元数据，获取 upload_id。
    """
    endpoint = pre["endpoint"]
    upos_uri = pre["upos_uri"].replace("upos:/", "")
    auth = pre["auth"]
    biz_id = pre["biz_id"]
    size = os.path.getsize(video_path)

    url = f"https:{endpoint}{upos_uri}"
    resp = session.post(
        url,
        params={
            "uploads": "",
            "output": "json",
            "profile": "ugcfx/bup",
            "filesize": size,
            "partsize": pre["chunk_size"],
            "biz_id": biz_id,
        },
        headers={"X-Upos-Auth": auth, "Content-Length": "0"}
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("OK") != 1:
        raise RuntimeError(f"元数据上传失败：{data}")
    return data


def upload_chunks(session: requests.Session, video_path: str, pre: dict, meta: dict, deadline: float = None) -> list:
    """
    分片上传视频，返回分片信息列表。
    """
    endpoint = pre["endpoint"]
    uri = pre["upos_uri"].replace("upos:/", "")
    auth = pre["auth"]
    chunk_size = pre["chunk_size"]
    upload_id = meta["upload_id"]
    total_size = os.path.getsize(video_path)
    total_chunks = math.ceil(total_size / chunk_size)
    url_base = f"https:{endpoint}{uri}"

    parts = []
    with open(video_path, "rb") as f:
        for i in range(total_chunks):
            # 动态计算剩余超时时间，防止单次网络请求死等卡住
            if deadline is not None:
                remaining_time = deadline - time.time()
                if remaining_time <= 0:
                    raise TimeoutError("分片上传超时：已超过设定的最大执行时间！")
                req_timeout = remaining_time
            else:
                req_timeout = None

            part = f.read(chunk_size)
            resp = session.put(
                url_base,
                params={
                    "partNumber": i + 1,
                    "uploadId": upload_id,
                    "chunk": i,
                    "chunks": total_chunks,
                    "size": len(part),
                    "start": i * chunk_size,
                    "end": i * chunk_size + len(part),
                    "total": total_size,
                },
                headers={"X-Upos-Auth": auth, "Content-Type": "application/octet-stream"},
                data=part,
                timeout=req_timeout  # 核心修改：将剩余时间赋给请求级别的 timeout
            )
            resp.raise_for_status()
            if resp.text.strip() != "MULTIPART_PUT_SUCCESS":
                raise RuntimeError(f"分片{i + 1}上传失败: {resp.text}")
            parts.append({"partNumber": i + 1, "eTag": "etag"})
    return parts


def finalize_upload(session: requests.Session, pre: dict, meta: dict, parts: list) -> dict:
    """
    完成上传合并视频。
    """
    endpoint = pre["endpoint"]
    uri = pre["upos_uri"].replace("upos:/", "")
    auth = pre["auth"]
    biz_id = pre["biz_id"]
    upload_id = meta["upload_id"]
    filename = os.path.basename(pre["upos_uri"])

    resp = session.post(
        f"https:{endpoint}{uri}",
        params={
            "output": "json",
            "name": filename,
            "profile": "ugcfx/bup",
            "uploadId": upload_id,
            "biz_id": biz_id,
        },
        headers={"X-Upos-Auth": auth, "Content-Type": "application/json"},
        json={"parts": parts},
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("OK") != 1:
        raise RuntimeError(f"结束上传失败：{data}")
    return data


def submit_post(
        session: requests.Session,
        cover_url: str,
        biz_id: int,
        filename: str,
        title: str,
        description: str,
        tags: str,
        copyright_type: int,
        tid: int,
        recreate: int,
        dynamic: str,
        no_reprint: int,
        bili_jct: str = BILI_JCT,
        human_type2=1002,
        topic_detail={"from_topic_id": 1313687, "from_source": "arc.web.recommend"},
        topic_id: int = 1313687

) -> dict:
    """
    投递稿件，返回 aid 和 bvid。
    """
    videos = [{"filename": filename, "title": "P1", "desc": "", "cid": biz_id}]
    payload = {
        "videos": videos,
        "cover": cover_url,
        "title": title,
        "copyright": copyright_type,
        "tid": tid,
        "tag": tags,
        "desc_format_id": 9999,
        "desc": description,
        "recreate": recreate,
        "dynamic": dynamic,
        "no_reprint": no_reprint,
        "interactive": 0,
        "act_reserve_create": 0,
        "no_disturbance": 0,
        "subtitle": {"open": 0, "lan": ""},
        "dolby": 0,
        "lossless_music": 0,
        "up_selection_reply": False,
        "up_close_reply": False,
        "up_close_danmu": False,
        "human_type2": human_type2,
        "topic_detail": topic_detail,
        "topic_id": topic_id,
        "web_os": 3,
        "csrf": bili_jct,
    }
    resp = session.post(
        "https://member.bilibili.com/x/vu/web/add/v3",
        params={"ts": int(time.time() * 1000), "csrf": bili_jct},
        headers={"Content-Type": "application/json; charset=utf-8"},
        json=payload,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != 0:
        raise RuntimeError(f"投递失败：{data.get('message')}")
    return data["data"]


def get_best_upcdn(upcdn_list: list, size=1) -> str:
    """
    优先选择没被使用的节点：
    1. 超过24小时未使用或从未测试过的节点。
    2. 如果都在24小时内，选择速度最快且 >= 100KB/s 的 CDN 节点。
    如果没有满足条件的节点，最多等待5分钟（有节点被释放且满足条件），超时抛错。
    增加死锁兜底机制：如果某个节点被占用超过1小时，强制释放加入可用列表。
    """
    file_size = size / 1024 / 1024  # 单位变成了MB
    cdn_info_path = r'W:\project\python_project\auto_video\config\cnd_info.json'
    lock_file_path = cdn_info_path + '.lock'

    SPEED_THRESHOLD = 50  # 100 KB/s
    wait_timeout = 1000  # 最大等待时间 300 秒 (5分钟)
    max_occupied_time = 1800
    start_wait_time = time.time()

    # 新增限制常量
    DEFAULT_SPEED_MB = 0.1  # 默认速度 0.1Mb/s
    MAX_EST_TIME = 900  # 最大允许预估时间 1800秒

    hour = datetime.now().hour
    if 1 <= hour < 6:
        MAX_EST_TIME = 3000

    while True:
        with SimpleFileLock(lock_file_path):
            cdn_info = read_json(cdn_info_path) if os.path.exists(cdn_info_path) else {}
            if not isinstance(cdn_info, dict):
                cdn_info = {}

            current_time = time.time()
            best_upcdn = None
            max_speed = -1

            # 1. 过滤出没被占用使用的节点，并处理死锁的节点
            available_cdns = []
            for cdn in upcdn_list:
                record = cdn_info.get(cdn, {})
                if record.get('occupied', False):
                    occupied_at = record.get('occupied_at', 0)
                    if current_time - occupied_at > max_occupied_time:
                        print(
                            f"发现节点 {cdn} 被占用超过1小时，判定为异常死锁，强制回收加入可用列表 当前时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
                        available_cdns.append(cdn)
                else:
                    available_cdns.append(cdn)

            if available_cdns:
                # 2. 检查是否有超过24小时未使用或完全没有记录的节点
                for cdn in available_cdns:
                    record = cdn_info.get(cdn, {})
                    if not record or (current_time - record.get("timestamp", 0) > 24 * 3600):
                        # 计算预计时间：无数据节点使用默认速度 0.1Mb
                        est_time = file_size / DEFAULT_SPEED_MB
                        if est_time > MAX_EST_TIME:
                            raise RuntimeError(
                                f"节点 {cdn} 预估上传时间 {est_time:.2f}s 超过最大限制 {MAX_EST_TIME}s (文件大小: {file_size:.2f}MB, 默认速度: {DEFAULT_SPEED_MB}MB/s)")

                        best_upcdn = cdn
                        print(
                            f"选择上传线路: {cdn} 线路节点速度 (无有效记录或记录过期) 预估上传时间 {est_time:.2f}s  当前时间：{time.strftime('%Y-%m-%d %H:%M:%S')}  file_size:{file_size}")
                        break

                # 3. 如果都在有效期内，选择速度最快且 >= 100KB/s 的节点
                if not best_upcdn:
                    for cdn in available_cdns:
                        record = cdn_info.get(cdn, {})
                        speed = record.get("speed", 0)
                        if speed >= SPEED_THRESHOLD and speed > max_speed:
                            max_speed = speed
                            best_upcdn = cdn

                    if best_upcdn:
                        # 计算预计时间：历史记录速度单位转换后计算
                        speed_mb = max_speed / 1024
                        # 防止除以0
                        est_time = file_size / speed_mb if speed_mb > 0 else file_size / DEFAULT_SPEED_MB
                        if est_time > MAX_EST_TIME:
                            raise RuntimeError(
                                f"节点 {best_upcdn} 预估上传时间 {est_time:.2f}s 超过最大限制 {MAX_EST_TIME}s (文件大小: {file_size:.2f}MB, 记录速度: {speed_mb:.2f}MB/s)")

                        print(
                            f"基于历史记录(24h内)，选择最快上传线路: 线路节点速度 {best_upcdn} (测速: {max_speed / 1024:.2f} MB/s) 预估上传时间 {est_time:.2f}s  当前时间：{time.strftime('%Y-%m-%d %H:%M:%S')} file_size:{file_size}")
                    else:
                        # 4. 新增兜底策略：只有在“所有节点都没被正常占用”（全局空闲）的情况下才兜底
                        # 如果 len(available_cdns) != len(upcdn_list)，说明还有节点在忙，应该继续循环等待
                        if len(available_cdns) == len(upcdn_list):
                            best_upcdn = min(available_cdns,
                                             key=lambda c: cdn_info.get(c, {}).get("timestamp", current_time))

                            record = cdn_info.get(best_upcdn, {})
                            speed = record.get("speed", 0)
                            speed_mb = speed / 1024 if speed > 0 else DEFAULT_SPEED_MB
                            est_time = file_size / speed_mb
                            print(f"触发兜底策略(全局无节点运行且速度均不达标)，强制选择更新最早的线路: {best_upcdn} (测速: {speed_mb:.2f} MB/s) 预估上传时间 {est_time:.2f}s  当前时间：{time.strftime('%Y-%m-%d %H:%M:%S')} file_size:{file_size}")

            # 如果找到了节点，标记为占用，并返回
            if best_upcdn:
                if best_upcdn not in cdn_info:
                    cdn_info[best_upcdn] = {"timestamp": current_time, "speed": 0}
                cdn_info[best_upcdn]['occupied'] = True
                cdn_info[best_upcdn]['occupied_at'] = current_time  # 记录具体的占用时间戳
                save_json(cdn_info_path, cdn_info)
                return best_upcdn

        # 如果未选到节点，检查是否超过 5 分钟
        if time.time() - start_wait_time > wait_timeout:
            raise RuntimeError("分配 CDN 节点超时：5 分钟内无满足条件的节点被释放。")

        # 短暂等待后重新尝试获取
        time.sleep(2)


def save_cdn_record(upcdn: str, file_size: int, duration: float):
    """
    计算上传速度（字节/秒）并保存到本地记录中。
    重要：无论时长是否有效，都必须将节点的占用的状态释放掉！
    新增逻辑：如果连续 3 次未产生有效测速（失败/中断），则将该节点测速强制重置为 0。
    """
    cdn_info_path = r'W:\project\python_project\auto_video\config\cnd_info.json'
    lock_file_path = cdn_info_path + '.lock'

    with SimpleFileLock(lock_file_path):
        try:
            cdn_info = read_json(cdn_info_path) if os.path.exists(cdn_info_path) else {}
            if not isinstance(cdn_info, dict):
                cdn_info = {}

            if upcdn not in cdn_info:
                cdn_info[upcdn] = {}

            # 核心机制：释放节点的占用状态
            cdn_info[upcdn]['occupied'] = False
            # occupied_at 不必清理，因为下一次被占用时会被新时间戳覆盖，且此处 occupied 已经是 False

            if duration > 0:
                speed = file_size / duration / 1024
                cdn_info[upcdn].update({
                    "size": file_size,
                    "duration": duration,
                    "speed": speed,
                    "timestamp": time.time(),
                    "fail_count": 0  # 成功产生测速时，清零连续失败次数
                })
                print(
                    f"成功记录并释放节点 {upcdn} 状态: 耗时 {duration:.2f}秒, 速度 {speed / 1024:.2f}MB/s 当前时间：{time.strftime('%Y-%m-%d %H:%M:%S')} ")
            else:
                # 失败或未产生有效测速时，累加连续失败次数
                fail_count = cdn_info[upcdn].get("fail_count", 0) + 1
                cdn_info[upcdn]["fail_count"] = fail_count

                if fail_count >= 5:
                    # 连续3次失败，强制将速度置为0，并更新时间戳（防止被当做"未测试"节点优先分配）
                    cdn_info[upcdn]["speed"] = 0
                    cdn_info[upcdn]["timestamp"] = time.time()
                    print(f"节点 {upcdn} 连续 {fail_count} 次未产生有效测速，已将测速强制置为 0 MB/s 当前时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
                else:
                    print(f"释放节点 {upcdn} (未产生有效测速，当前连续失败 {fail_count} 次) 当前时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")

            save_json(cdn_info_path, cdn_info)
        except Exception as e:
            print(f"记录/释放 CDN 状态时发生错误: {e}")

def upload_to_bilibili(
        video_path: str,
        cover_path: str,
        title: str,
        description: str,
        tags: str,
        copyright_type: int = DEFAULT_COPYRIGHT,
        tid: int = DEFAULT_TID,
        recreate: int = DEFAULT_RECREATE,
        dynamic: str = DEFAULT_DYNAMIC,
        no_reprint: int = DEFAULT_NO_REPRINT,
        sessdata=SESSDATA,
        bili_jct=BILI_JCT,
        full_cookie=full_cookie,
        human_type2=1002,
        topic_detail={"from_topic_id": 1313687, "from_source": "arc.web.recommend"},
        topic_id: int = 1313687,
        max_execution_time: int = 1800
) -> dict:
    """
    一步完成B站投稿流程，返回投稿结果。
    """
    start_time = time.time()
    deadline = start_time + max_execution_time

    if not all([video_path, cover_path, title, description, tags]):
        raise ValueError("缺少必要参数：视频路径/封面路径/标题/简介/标签")
    if not os.path.exists(video_path) or not os.path.exists(cover_path):
        raise FileNotFoundError("视频或封面文件不存在")

    sess = get_session(sessdata, bili_jct, full_cookie)

    def check_timeout(step_name):
        if time.time() > deadline:
            raise TimeoutError(f"【{step_name}】时已超过最大执行时间限制（{max_execution_time}秒），自动终止。")

    cover_url = upload_cover(sess, cover_path, bili_jct=bili_jct)
    check_timeout("上传封面")



    # 将上传逻辑使用 try-finally 包裹，保证不论成功失败，节点占用必定被释放
    try:
        pre = preupload_video(sess, video_path)
        check_timeout("预上传视频")
        video_upload_start_time = time.time()  # 记录纯视频上传开始时间

        biz_id = pre["biz_id"]
        filename = os.path.splitext(os.path.basename(pre["upos_uri"]))[0]
        chosen_upcdn = pre.pop('_chosen_upcdn', None)  # 提取 upcdn

        video_upload_end_time = None

        meta = post_video_meta(sess, pre, video_path)
        check_timeout("提交视频元数据")

        parts = upload_chunks(sess, video_path, pre, meta, deadline=deadline)
        check_timeout("分片上传")

        finalize_upload(sess, pre, meta, parts)
        check_timeout("完成上传合并")

        video_upload_end_time = time.time()  # 记录纯视频上传结束时间

    finally:
        # --- 调用独立的保存函数去记录速度并释放被占用的节点 ---
        if chosen_upcdn:
            if video_upload_end_time:
                upload_duration = video_upload_end_time - video_upload_start_time
                video_size = os.path.getsize(video_path)
                save_cdn_record(chosen_upcdn, video_size, upload_duration)
            else:
                # 说明中间上传出现了中断或失败，传入 0 确保只释放节点不更新错误测速
                save_cdn_record(chosen_upcdn, 0, 0)

    return submit_post(
        sess, cover_url, biz_id, filename,
        title, description, tags,
        copyright_type, tid, recreate, dynamic, no_reprint,
        bili_jct=bili_jct, human_type2=human_type2, topic_detail=topic_detail, topic_id=topic_id
    )


def get_bili_category_id(cookie_str):
    """
    发起与下面 fetch 等效的 GET 请求，并只返回解析后的响应内容（JSON -> dict/list，否则返回文本）。

    参数:
        cookie_str: 浏览器格式的 Cookie 字符串，例如 "SESSDATA=xxx; bili_jct=yyy; ..."

    返回:
        解析后的响应（JSON -> dict/list；否则返回字符串）

    抛出:
        requests.HTTPError: 当返回非 2xx 状态码时
        requests.RequestException: 网络、超时等请求错误
    """
    url = "https://member.bilibili.com/x/vupre/web/archive/human/type2/list?t=1767616562119"

    headers = {
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Priority": "u=1, i",
        "Sec-CH-UA": "\"Google Chrome\";v=\"143\", \"Chromium\";v=\"143\", \"Not A(Brand\";v=\"24\"",
        "Sec-CH-UA-Mobile": "?0",
        "Sec-CH-UA-Platform": "\"Windows\"",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "Referer": "https://member.bilibili.com/platform/upload/video/frame?spm_id_from=333.1007.top_bar.upload",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36",
        "Cookie": cookie_str,
    }

    # 发起请求（会在网络错误/超时时抛出 requests.RequestException）
    resp = requests.get(url, headers=headers, timeout=10)

    # 非 2xx 直接抛出 HTTPError，调用方可捕获并处理
    resp.raise_for_status()

    # 尝试解析为 JSON，否则返回文本
    try:
        return resp.json()
    except ValueError:
        return resp.text


def send_bilibili_dm_command(cookie_str, csrf, question, duration, optionA, optionB, bvid, aid):
    """
    """

    try:
        url = "https://api.bilibili.com/x/v2/dm/command/post"
        cid = get_cid_from_bvid(bvid)

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:147.0) Gecko/20100101 Firefox/147.0",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "zh-CN,zh;q=0.9,zh-TW;q=0.8,zh-HK;q=0.7,en-US;q=0.6,en;q=0.5",
            "Content-Type": "application/x-www-form-urlencoded",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-site",
            "Referer": "https://www.bilibili.com/",
            "Cookie": cookie_str
        }

        # Prepare the 'data' JSON payload
        data_payload = {
            "question": question,
            "duration": duration,
            "posX_2": 97,
            "posY_2": 85,
            "pub_dynamic": False,
            "has_self_def": "0",
            "optionA": optionA,
            "optionB": optionB
        }
        data_str = json.dumps(data_payload, ensure_ascii=False)

        # Form data for the POST request
        form_data = {
            "type": "9",
            "aid": aid,
            "cid": cid,
            "progress": "0",
            "plat": "8",
            "data": data_str,
            "csrf": csrf
        }

        response = requests.post(url, headers=headers, data=form_data)
        response.raise_for_status()
        print(f"DM command sent successfully: {response.text}")
        return response.json()
    except Exception as e:
        traceback.print_exc()
        print(f"Error sending DM command: {e}")
        return None


if __name__ == "__main__":
    # config_map = init_config()
    # user_name = 'mama'
    # target_value = None
    # for uid, value in config_map.items():
    #     if value.get('name') == user_name:
    #         target_value = value
    #         break
    # total_cookie = target_value.get('total_cookie')
    # BILI_JCT = target_value.get('BILI_JCT')
    # send_bilibili_dm_command(total_cookie, BILI_JCT, "这是一个测试问题？", 5000, "选项A", "选项B", "BV1Y6fPBmEvk", "116116487668454")

    #
    # # 读取LLM/TikTokDownloader/metadata_cache.json
    # with open('../../LLM/TikTokDownloader/metadata_cache.json', 'r', encoding='utf-8') as f:
    #     metadata_cache = json.load(f)
    # for key, value in metadata_cache.items():
    #     video_path = value.get('video_path')
    #
    start_time = time.time()
    result = upload_to_bilibili(
        video_path=r"W:\project\python_project\auto_video\utils\test.mp4",
        cover_path=r'W:\project\python_project\auto_video\utils\test.jpg',
        title="我的AI修复视频与精彩瞬间",
        description="这是一个使用AI技术修复后的视频，并加入了有趣的音频，希望大家喜欢！",
        tags="AI修复,视频剪辑,有趣,科技,日常生活",
    )
    print(f"总耗时: {time.time() - start_time:.2f}秒")
    # print(f"投稿成功！AID={result['aid']}, BVID={result['bvid']}")
"""
视频处理与发布任务管理服务

该模块提供Flask Web服务，用于接收前端提交的视频处理任务，
解析抖音视频元数据，并将任务信息存储到MongoDB中。
重构版：函数式编程风格，扁平化结构，逻辑解耦。
包含优化：
1. 优先利用本地映射和数据库缓存，减少重复解析。
2. 任务查重加入 creative_guidance 校验，支持同素材不同指令的多次生成。
3. 确保所有涉及的素材（无论新旧）都会再次执行保存操作。
4. [新增] 统一返回格式 {status, message, errors}。
5. [新增] 批次内 video_id 查重，防止单次请求包含重复视频内容。
6. [修复] 修复了 build_video_material_data 返回值解包错误，并正确集成了 validate_timestamp。
"""

import time
from typing import Optional, List, Tuple, Dict, Any

from flask import Flask, request, jsonify, render_template, Response

from utils.common_utils import read_json, save_json, check_timestamp
# 导入配置和工具
from video_common_config import TaskStatus, _configure_third_party_paths, ErrorMessage, ResponseStatus, \
    ALLOWED_USER_LIST, LOCAL_ORIGIN_URL_ID_INFO_PATH, fix_split_time_points

_configure_third_party_paths()

from utils.mongo_base import gen_db_object
from utils.mongo_manager import MongoManager
from third_party.TikTokDownloader.douyin_downloader import get_meta_info

app = Flask(__name__)


# =============================================================================
# 0. 基础设施初始化
# =============================================================================

def _init_mongo_manager() -> MongoManager:
    """初始化MongoDB管理器"""
    print("Initializing MongoDB connection...")
    mongo_base_instance = gen_db_object()
    manager = MongoManager(mongo_base_instance)
    print("✅ MongoDB Manager is ready.")
    return manager


# 全局数据库管理器实例
mongo_manager = _init_mongo_manager()


# =============================================================================
# 1. 纯逻辑函数层 (数据解析与构建) - 无状态，只负责处理数据
# =============================================================================

def parse_douyin_video(video_url: str):
    """
    解析单个抖音视频URL
    Returns: (是否成功, 元数据字典, 错误信息)
    """
    try:
        meta_data_list = get_meta_info(video_url)
        if not meta_data_list:
            return False, None, ErrorMessage.PARSE_NO_METADATA
        return True, meta_data_list[0], None
    except Exception as e:
        print(f"解析URL '{video_url}' 异常: {e}")
        return False, None, f"解析异常: {e}"

def validate_timestamp(video_item, duration):
    """
    校验用户输入的时间戳是否超出视频时长
    """
    remove_time_segments = video_item.get('remove_time_segments', [])
    split_time_points = video_item.get('split_time_points', [])

    # 这里的切片逻辑需要拷贝一份，防止修改原数据
    all_timestamps = list(split_time_points)
    for remove_time_segment in remove_time_segments:
        # 简单校验格式，防止 crash
        if '-' in remove_time_segment:
            parts = remove_time_segment.split('-')
            if len(parts) == 2:
                all_timestamps.append(parts[0])
                all_timestamps.append(parts[1])

    # 调用 common_utils 中的校验逻辑
    error_info = check_timestamp(all_timestamps, duration)
    return error_info, None

def build_video_material_data(video_item: Dict, meta_data: Dict, video_id: str):
    """构建单条视频素材的存库数据结构"""
    # 提取基础信息
    base_info = {
        'video_title': meta_data.get('full_title') or meta_data.get('desc'),
        'video_desc': meta_data.get('desc'),
        'collection_time': meta_data.get('collection_time'),
        'author': meta_data.get('nickname'),
        'upload_time': meta_data.get('create_time'),
        'duration': meta_data.get('duration'),
        'tags': meta_data.get('text_extra', []),
        'height': meta_data.get('height'),
        'width': meta_data.get('width'),
        'original_url': video_item.get('original_url'),
        'download_url': meta_data.get('downloads'),
        'dynamic_cover': meta_data.get('dynamic_cover'),
        'static_cover': meta_data.get('static_cover'),
        'digg_count': meta_data.get('digg_count'),
        'comment_count': meta_data.get('comment_count'),
        'collect_count': meta_data.get('collect_count'),
        'share_count': meta_data.get('share_count'),
        'comment_list': []
    }

    return {
        'video_id': video_id,
        'status': TaskStatus.PROCESSING,
        'error_info': None,
        'base_info': base_info,
        'extra_info': video_item
    }


def build_publish_task_data(user_name: str, global_settings: Dict, materials: List[Dict],
                            original_video_list: List[Dict]) -> Dict:
    """构建发布任务的存库数据结构"""
    video_id_list = [m['video_id'] for m in materials]

    # 构建 original_url_info_list (将原始URL映射到解析后的video_id)
    successful_urls = {}
    for m in materials:
        original_url = m.get('base_info', {}).get('original_url')
        if original_url:
            successful_urls[original_url] = m['video_id']

    url_info_list = []

    for item in original_video_list:
        url = item.get('original_url')
        if url in successful_urls:
            info_item = item.copy()
            info_item['video_id'] = successful_urls[url]
            url_info_list.append(info_item)

    return {
        'video_id_list': video_id_list,
        'userName': user_name,
        'status': TaskStatus.PROCESSING,
        'failed_count': 0,
        'original_url_info_list': url_info_list,
        'creation_guidance_info': global_settings,
        'new_video_script_info': None,
        'upload_info': None
    }


# =============================================================================
# 2. 业务流程控制层 - 串联逻辑与数据库
# =============================================================================

def process_one_click_generate(request_data: Dict) -> Tuple[Dict, int]:
    """
    处理一键生成请求的核心业务流程
    """
    # 初始化返回结构
    response_structure = {
        'status': ResponseStatus.ERROR,
        'message': '',
        'errors': []  # 明确定义 errors 字段
    }

    # --- Step 1: 基础参数校验 ---
    if not request_data or not request_data.get('userName') or not request_data.get('video_list'):
        response_structure['message'] = ErrorMessage.MISSING_REQUIRED_FIELDS
        response_structure['errors'].append("缺少 userName 或 video_list 参数")
        return response_structure, 400

    user_name = request_data['userName']
    if user_name not in ALLOWED_USER_LIST:
        response_structure['message'] = f"用户名{user_name}未向管理员注册"
        response_structure['errors'].append("用户鉴权失败")
        return response_structure, 403

    global_settings = request_data.get('global_settings', {})
    input_video_list = request_data['video_list']

    print(f"开始处理请求 | 用户: {user_name} | 视频数: {len(input_video_list)}")

    # --- Step 2: 获取视频素材 (优先查本地缓存+DB，失败则解析) ---
    valid_materials = []
    errors = []
    # [新增] 用于当前批次的去重集合
    current_batch_video_ids = set()

    original_url_id_info = read_json(LOCAL_ORIGIN_URL_ID_INFO_PATH)
    is_url_mapping_updated = False

    for idx, video_item in enumerate(input_video_list, start=1):
        url = video_item.get('original_url', '').strip()
        if not url:
            errors.append(f"第 {idx} 条记录错误: 视频链接为空")
            continue

        cached_material = None
        local_video_id = original_url_id_info.get(url)
        meta_data = None # 临时存储元数据
        current_video_id = None # 临时存储 ID
        duration = "00:01" # 临时存储时长

        # 1. 尝试从缓存获取
        if local_video_id:
            # 数据库查询
            db_results = mongo_manager.find_materials_by_ids([local_video_id])
            if db_results and len(db_results) > 0:
                print(f"URL命中缓存，跳过解析: {url}")
                cached_material = db_results[0]
                # 更新 input 的指令到 material 中
                cached_material['extra_info'] = video_item
                current_video_id = cached_material.get('video_id')
                # 从缓存中获取时长
                duration = cached_material.get('base_info', {}).get('duration')

        # 2. 如果缓存未命中，执行解析
        if not cached_material:
            print(f"缓存未命中，开始解析: {url}")
            success, meta, err_msg = parse_douyin_video(url)

            if not success:
                errors.append(f"第 {idx} 条记录解析失败 (URL: {url}): {err_msg}")
                continue

            current_video_id = meta.get('id')
            if not current_video_id:
                errors.append(f"第 {idx} 条记录解析成功但无ID (URL: {url})")
                continue

            # 解析成功，记录元数据和时长
            meta_data = meta
            duration = meta.get('duration', 0)

            # 标记需要更新本地映射
            original_url_id_info[url] = current_video_id
            is_url_mapping_updated = True

        # --- 核心修改：在此处插入 validate_timestamp ---
        # 此时无论来自缓存还是新解析，我们都有了 duration 和 video_item
        time_err, _ = validate_timestamp(video_item, duration)
        if time_err:
            errors.append(f"第 {idx} 条记录时间戳错误 (URL: {url}): {time_err}")
            continue
        # -------------------------------------------

        # 3. 如果是新解析的数据，构建 material 对象 (缓存命中的话 cached_material 已存在)
        if not cached_material:
            # 注意：build_video_material_data 返回的是 dict，不是 tuple
            cached_material = build_video_material_data(video_item, meta_data, current_video_id)

        # 4. [新增] 核心逻辑：检查当前批次是否重复
        if current_video_id in current_batch_video_ids:
            errors.append(f"第 {idx} 条记录重复 (URL: {url}): 对应的视频ID {current_video_id} 已存在于当前提交的任务列表中，不允许重复添加。")
            continue

        # 通过所有检查，加入列表
        current_batch_video_ids.add(current_video_id)
        if cached_material:
            valid_materials.append(cached_material)

    # 回写本地缓存
    if is_url_mapping_updated:
        save_json(LOCAL_ORIGIN_URL_ID_INFO_PATH, original_url_id_info)

    # 如果有任何错误（包括解析失败或重复），根据业务需求这里选择报错返回
    if errors:
        response_structure['message'] = ErrorMessage.PARTIAL_PARSE_FAILURE
        response_structure['errors'] = errors
        return response_structure, 400

    if not valid_materials:
        response_structure['message'] = "无有效视频可处理"
        response_structure['errors'].append("解析后未获得任何有效素材")
        return response_structure, 400

    # --- Step 3: 任务查重 (历史任务校验) ---
    video_ids = [m['video_id'] for m in valid_materials]
    existing_tasks = mongo_manager.find_task_by_exact_video_ids(video_ids)

    is_duplicate_task = False
    current_guidance = global_settings.get('creative_guidance', '')

    if existing_tasks:
        if not isinstance(existing_tasks, list):
            existing_tasks = [existing_tasks]

        for task in existing_tasks:
            task_guidance_info = task.get('creation_guidance_info', {}) or {}
            old_guidance = task_guidance_info.get('creative_guidance', '')
            if old_guidance == current_guidance:
                is_duplicate_task = True
                break

    if is_duplicate_task:
        print(f"用户 {user_name} 提交的任务完全重复，跳过。")
        response_structure['status'] = ResponseStatus.SUCCESS # 业务上算成功处理（已存在）
        response_structure['message'] = ErrorMessage.TASK_ALREADY_EXISTS
        response_structure['errors'] = [] # 无错误，只是提示已存在
        return response_structure, 200

    # --- Step 4 & 5: 保存数据 ---
    try:
        mongo_manager.upsert_materials(valid_materials)
        task_data = build_publish_task_data(user_name, global_settings, valid_materials, input_video_list)
        mongo_manager.upsert_tasks([task_data])

        print(f"成功创建新任务，包含 {len(valid_materials)} 个视频。")
        response_structure['status'] = ResponseStatus.SUCCESS
        response_structure['message'] = f'新任务已成功创建，包含 {len(valid_materials)} 个视频。'
        response_structure['errors'] = [] # 成功时 errors 为空列表
        return response_structure, 200

    except Exception as e:
        app.logger.error(f"数据库操作失败: {e}")
        response_structure['message'] = "系统内部错误: 数据库保存失败"
        response_structure['errors'].append(str(e))
        return response_structure, 500


# =============================================================================
# 3. Flask 路由接口层
# =============================================================================

@app.route('/')
def index() -> str:
    return render_template('index.html')


@app.route('/one-click-generate', methods=['POST'])
def one_click_generate() -> Tuple[Response, int]:
    """一键生成接口"""
    try:
        data = request.get_json()
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 收到请求: {data}")

        response_data, status_code = process_one_click_generate(data)

        return jsonify(response_data), status_code

    except Exception as e:
        app.logger.exception("one_click_generate 接口发生未处理异常")
        return jsonify({
            'status': ResponseStatus.ERROR,
            'message': '内部服务器错误',
            'errors': [str(e)]
        }), 500


@app.route('/get_user_upload_info', methods=['GET'])
def get_user_upload_info() -> Response:
    """获取用户上传统计信息 (Mock)"""
    return jsonify({
        'status': ResponseStatus.SUCCESS,
        'message': '获取成功',
        'errors': [],
        'data': {
            'total_count_today': 0,
            'unprocessed_count_today': 0,
            'remote_upload_count': 0
        }
    })


if __name__ == "__main__":
    print("Flask 接口服务启动...")
    app.run(host='0.0.0.0', port=5002, debug=True, use_reloader=False)
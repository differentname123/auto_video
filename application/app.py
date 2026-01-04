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
7. [重构] 全面优化 process_one_click_generate 逻辑，分离解析、DB查询与校验逻辑。
"""

import time
from typing import Optional, List, Tuple, Dict, Any, Set

from flask import Flask, request, jsonify, render_template, Response

from application.process_video import query_need_process_tasks
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
# 1. 纯逻辑函数层 (数据解析与构建)
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


def check_cached_material(cached_material, video_item):
    """
    判断单个视频传入的信息是否改变。
    比较 db 中的 extra_info 和 传入的 video_item，
    但排除 '_cardDomId', 'original_url', 'is_realtime_video' 这几个字段。
    """
    # 1. 定义不需要比较的字段集合
    ignore_keys = {'_cardDomId', 'original_url', 'is_realtime_video'}

    # 2. 获取 DB 中的数据，处理可能为 None 的情况
    db_info = cached_material.get('extra_info') or {}

    # 3. 获取传入的数据，处理可能为 None 的情况
    current_info = video_item or {}

    # 4. 生成过滤后的字典（只包含未被忽略的字段）
    # 使用字典推导式：遍历原字典，只有 key 不在 ignore_keys 中才保留
    clean_db_info = {k: v for k, v in db_info.items() if k not in ignore_keys}
    clean_current_info = {k: v for k, v in current_info.items() if k not in ignore_keys}

    is_same = clean_db_info == clean_current_info

    if not is_same:
        tasks_to_process = query_need_process_tasks()
        # 统计所有的 video_id
        video_ids_in_process = set()
        for task in tasks_to_process:
            video_ids_in_process.update(task.get('video_id_list', []))
        # 如果在这个列表中，强制返回 False，避免覆盖
        if cached_material.get('video_id') in video_ids_in_process:
            return False

        print(f"{cached_material.get('video_id')} 检测到素材配置变更，清理相关缓存数据...")
        cached_material['logical_scene_info'] = None
        cached_material['video_overlays_text_info'] = None
        cached_material['owner_asr_info'] = None
        cached_material['hudong_info'] = None


    return True

# =============================================================================
# 2. 业务流程辅助函数 (解耦逻辑)
# =============================================================================

def _validate_request_basic(request_data: Dict) -> Tuple[bool, List[str]]:
    """Step 1: 基础参数校验"""
    errors = []
    if not request_data or not request_data.get('userName') or not request_data.get('video_list'):
        errors.append("缺少 userName 或 video_list 参数")
        return False, errors

    user_name = request_data['userName']
    if user_name not in ALLOWED_USER_LIST:
        errors.append(f"用户鉴权失败: {user_name} 未注册")
        return False, errors

    return True, errors

def _resolve_video_ids(input_video_list: List[Dict]) -> Tuple[Dict[str, str], Dict[str, Dict], List[str]]:
    """
    Step 2: 解析所有URL获取 video_id 和 meta_data

    Returns:
        url_to_id_map: URL -> video_id
        id_to_meta_map: video_id -> new_parsed_meta_data (本地有缓存则无此数据)
        errors: 错误信息列表
    """
    original_url_id_info = read_json(LOCAL_ORIGIN_URL_ID_INFO_PATH)
    is_url_mapping_updated = False

    url_to_id_map = {}
    id_to_meta_map = {} # 仅存储新解析的 meta
    errors = []

    # 请求级缓存，防止单次请求中包含重复URL时重复解析
    request_scope_parsed_cache = {}

    for idx, video_item in enumerate(input_video_list, start=1):
        url = video_item.get('original_url', '').strip()
        if not url:
            errors.append(f"第 {idx} 条记录错误: 视频链接为空")
            continue

        # 优先从本地文件映射获取
        local_video_id = original_url_id_info.get(url)

        if local_video_id:
            url_to_id_map[url] = local_video_id
            continue

        # 其次检查本次请求内是否已经解析过该URL
        if url in request_scope_parsed_cache:
            success, meta, err_msg = request_scope_parsed_cache[url]
        else:
            # 否则进行网络解析 (耗时操作)
            print(f"本地缓存未命中，执行解析: {url}")
            success, meta, err_msg = parse_douyin_video(url)
            request_scope_parsed_cache[url] = (success, meta, err_msg)

        if not success:
            errors.append(f"第 {idx} 条记录解析失败 (URL: {url}): {err_msg}")
            continue

        current_video_id = meta.get('id')
        if not current_video_id:
            errors.append(f"第 {idx} 条记录解析成功但无ID (URL: {url})")
            continue

        # 记录映射和元数据
        url_to_id_map[url] = current_video_id
        id_to_meta_map[current_video_id] = meta

        # 更新本地映射表
        original_url_id_info[url] = current_video_id
        is_url_mapping_updated = True

    if is_url_mapping_updated:
        save_json(LOCAL_ORIGIN_URL_ID_INFO_PATH, original_url_id_info)

    return url_to_id_map, id_to_meta_map, errors

def _fetch_and_map_db_materials(video_ids: List[str]) -> Dict[str, Dict]:
    """Step 3: 根据 ID 列表批量查询数据库，返回 {video_id: material} 映射"""
    if not video_ids:
        return {}

    # 需求1：对所有的 video_id 进行查询，无论来源是本地缓存还是新解析
    db_results = mongo_manager.find_materials_by_ids(video_ids)
    return {m['video_id']: m for m in db_results}

def _process_material_construction(input_video_list: List[Dict],
                                 url_to_id_map: Dict[str, str],
                                 id_to_meta_map: Dict[str, Dict],
                                 db_materials_map: Dict[str, Dict]) -> Tuple[List[Dict], List[str]]:
    """
    Step 4: 整合数据，构建或校验 Material 对象

    Returns:
        valid_materials: 待保存的素材列表
        errors: 处理过程中的错误
    """
    valid_materials = []
    errors = []
    # 需求5：批次内查重
    current_batch_video_ids = set()

    for idx, video_item in enumerate(input_video_list, start=1):
        url = video_item.get('original_url', '').strip()
        if not url:
            continue # 已在 Step 2 报错，这里跳过

        video_id = url_to_id_map.get(url)
        if not video_id:
            continue # 解析失败的跳过

        # 批次内重复校验
        if video_id in current_batch_video_ids:
            errors.append(f"第 {idx} 条记录重复 (URL: {url}): ID {video_id} 已在当前任务列表中")
            continue


        # A. 检查数据库中是否存在 (需求1：优先以DB为准，防止配置覆盖)
        if video_id in db_materials_map:
            db_material = db_materials_map[video_id]

            # 校验配置是否变更
            if not check_cached_material(db_material, video_item):
                errors.append(f"第 {idx} 条素材配置不允许修改，因为还有使用该素材的视频创作任务未完成 (URL: {url})")
                continue

            # 使用DB中的时长进行校验
            duration = db_material.get('base_info', {}).get('duration')

            # 复用DB对象，但更新 extra_info (虽然上面校验了一致性，这里赋值是为了保持逻辑统一)
            db_material['extra_info'] = video_item
            final_material = db_material

        # B. 数据库无记录，使用新解析的数据
        elif video_id in id_to_meta_map:
            meta_data = id_to_meta_map[video_id]
            duration = meta_data.get('duration', 0)
            final_material = build_video_material_data(video_item, meta_data, video_id)

        else:
            # 理论上不应该走到这里，除非 ID 既不在 DB 也不在 Meta Map
            errors.append(f"第 {idx} 条记录数据异常，无法获取元数据 (URL: {url})")
            continue

        # C. 统一校验时间戳
        time_err, _ = validate_timestamp(video_item, duration)
        if time_err:
            errors.append(f"第 {idx} 条记录时间戳错误 (URL: {url}): {time_err}")
            continue

        # D. 加入结果集
        current_batch_video_ids.add(video_id)
        valid_materials.append(final_material)

    return valid_materials, errors

def _check_task_duplication(user_name: str, valid_materials: List[Dict], global_settings: Dict) -> bool:
    """Step 5: 检查任务是否完全重复"""
    video_ids = [m['video_id'] for m in valid_materials]
    existing_tasks = mongo_manager.find_task_by_exact_video_ids(video_ids)

    current_guidance = global_settings.get('creative_guidance', '')

    if existing_tasks:
        if not isinstance(existing_tasks, list):
            existing_tasks = [existing_tasks]

        for task in existing_tasks:
            task_user_name = task.get('userName', '')
            task_guidance_info = task.get('creation_guidance_info', {}) or {}
            old_guidance = task_guidance_info.get('creative_guidance', '')
            # 如果素材列表完全一致，且创作指导也一致，则认为是重复任务
            if old_guidance == current_guidance and task_user_name == user_name:
                return True
    return False

# =============================================================================
# 3. 核心业务流程 (重构后)
# =============================================================================

def process_one_click_generate(request_data: Dict) -> Tuple[Dict, int]:
    """
    处理一键生成请求的核心业务流程
    """
    response_structure = {
        'status': ResponseStatus.ERROR,
        'message': '',
        'errors': []
    }

    # 1. 基础参数校验
    is_valid_req, req_errors = _validate_request_basic(request_data)
    if not is_valid_req:
        response_structure['message'] = ErrorMessage.MISSING_REQUIRED_FIELDS
        response_structure['errors'] = req_errors
        return response_structure, 400

    user_name = request_data['userName']
    input_video_list = request_data['video_list']
    global_settings = request_data.get('global_settings', {})

    print(f"开始处理请求 | 用户: {user_name} | 视频数: {len(input_video_list)}")

    # 2. 解析 ID (优先缓存，其次网络) - 需求3
    url_to_id_map, id_to_meta_map, parse_errors = _resolve_video_ids(input_video_list)
    if parse_errors:
        response_structure['message'] = ErrorMessage.PARTIAL_PARSE_FAILURE
        response_structure['errors'] = parse_errors
        return response_structure, 400

    # 3. 批量查询数据库 - 需求1
    # 获取所有相关的 video_id 对应的 DB 记录
    all_resolved_ids = list(url_to_id_map.values())
    db_materials_map = _fetch_and_map_db_materials(all_resolved_ids)

    # 4. 构建与校验素材对象
    valid_materials, build_errors = _process_material_construction(
        input_video_list, url_to_id_map, id_to_meta_map, db_materials_map
    )

    if build_errors:
        response_structure['message'] = "素材校验未通过"
        response_structure['errors'] = build_errors
        return response_structure, 400

    if not valid_materials:
        response_structure['message'] = "无有效视频可处理"
        return response_structure, 400

    # 5. 任务查重
    if _check_task_duplication(user_name, valid_materials, global_settings):
        print(f"用户 {user_name} 提交的任务完全重复，跳过。")
        response_structure['status'] = ResponseStatus.SUCCESS
        response_structure['message'] = ErrorMessage.TASK_ALREADY_EXISTS
        response_structure['errors'] = ['可尝试采用不同的素材或者调整创作指导也能创建新任务']
        return response_structure, 500

    # 6. 保存数据
    try:
        mongo_manager.upsert_materials(valid_materials)
        task_data = build_publish_task_data(user_name, global_settings, valid_materials, input_video_list)
        mongo_manager.upsert_tasks([task_data])

        print(f"成功创建新任务，包含 {len(valid_materials)} 个视频。")
        response_structure['status'] = ResponseStatus.SUCCESS
        response_structure['message'] = f'新任务已成功创建，包含 {len(valid_materials)} 个视频。'
        return response_structure, 200

    except Exception as e:
        app.logger.error(f"数据库操作失败: {e}")
        response_structure['message'] = "系统内部错误: 数据库保存失败"
        response_structure['errors'].append(str(e))
        return response_structure, 500


def process_check_video_status(request_data: Dict) -> Tuple[Dict, int]:
    """
    检查视频列表的状态：
    1. 解析 URL 获取 video_id
    2. 检查 DB 是否存在该 video_id
    3. 如果存在，返回 DB 中的配置信息供前端同步
    """
    if not request_data or not request_data.get('video_list'):
        return {'status': ResponseStatus.ERROR, 'message': '参数缺失', 'errors': []}, 400

    input_video_list = request_data['video_list']
    original_url_id_info = read_json(LOCAL_ORIGIN_URL_ID_INFO_PATH)
    is_url_mapping_updated = False

    check_results = []

    # 复用 _resolve_video_ids 中的部分逻辑，但这里我们需要逐个构建 result，所以保留原结构稍微优化

    for idx, video_item in enumerate(input_video_list):
        url = video_item.get('original_url', '').strip()
        if not url:
            check_results.append({'index': idx, 'status': 'error', 'msg': 'URL为空'})
            continue

        current_video_id = original_url_id_info.get(url)

        # 1. 尝试获取 video_id (缓存 -> 解析)
        if not current_video_id:
            success, meta, err_msg = parse_douyin_video(url)
            if success:
                current_video_id = meta.get('id')
                original_url_id_info[url] = current_video_id
                is_url_mapping_updated = True
            else:
                check_results.append({'index': idx, 'status': 'error', 'msg': f'解析失败: {err_msg}'})
                continue

        # 2. 查询数据库
        db_results = mongo_manager.find_materials_by_ids([current_video_id])

        if db_results and len(db_results) > 0:
            # 找到已有素材，提取配置信息
            existing_material = db_results[0]
            stored_config = existing_material.get('extra_info', {})

            check_results.append({
                'index': idx,
                'status': 'exists',
                'video_id': current_video_id,
                'original_url': url,
                'stored_config': stored_config,
                'msg': '发现历史配置'
            })
        else:
            # 这是一个全新的视频
            check_results.append({
                'index': idx,
                'status': 'new',
                'video_id': current_video_id,
                'msg': '新素材'
            })

    if is_url_mapping_updated:
        save_json(LOCAL_ORIGIN_URL_ID_INFO_PATH, original_url_id_info)

    return {
        'status': ResponseStatus.SUCCESS,
        'data': check_results,
        'message': '检查完成'
    }, 200


# =============================================================================
# 4. Flask 路由接口层
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


@app.route('/check-video-status', methods=['POST'])
def check_video_status() -> Tuple[Response, int]:
    """检查视频状态接口"""
    try:
        data = request.get_json()
        response_data, status_code = process_check_video_status(data)
        return jsonify(response_data), status_code
    except Exception as e:
        app.logger.exception("check_video_status 接口异常")
        return jsonify({'status': 'error', 'message': str(e)}), 500


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
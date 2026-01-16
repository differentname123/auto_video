import time
import traceback
import multiprocessing
import threading  # [新增] 用于后台运行监控循环
from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Dict, Any, Set

from flask import Flask, request, jsonify, render_template, Response

from application.process_video import query_need_process_tasks, _task_process_worker, _task_producer_worker, \
    check_task_queue
from utils.common_utils import read_json, save_json, check_timestamp, delete_files_in_dir_except_target, get_user_type
# 导入配置和工具
from video_common_config import TaskStatus, _configure_third_party_paths, ErrorMessage, ResponseStatus, \
    ALLOWED_USER_LIST, LOCAL_ORIGIN_URL_ID_INFO_PATH, fix_split_time_points, build_video_paths, \
    USER_STATISTIC_INFO_PATH, STATISTIC_PLAY_COUNT_FILE, VIDEO_MAX_RETRY_TIMES

_configure_third_party_paths()

from utils.mongo_base import gen_db_object
from utils.mongo_manager import MongoManager
from third_party.TikTokDownloader.douyin_downloader import get_meta_info

app = Flask(__name__)

# =============================================================================
# 0. 全局多进程共享对象 (新增部分)
# =============================================================================
# 定义全局变量，以便在 Flask 视图函数中访问
global_manager = None
running_task_ids = None # 共享去重字典 (Key: video_id)
task_queue = None       # 任务队列
consumers = []          # 消费者进程列表
producer_p = None       # 生产者进程


def _init_mongo_manager() -> MongoManager:
    """初始化MongoDB管理器"""
    # print("Initializing MongoDB connection...")
    mongo_base_instance = gen_db_object()
    manager = MongoManager(mongo_base_instance)
    # print("✅ MongoDB Manager is ready.")
    return manager


# 全局数据库管理器实例
mongo_manager = _init_mongo_manager()


# =============================================================================
# 1. 纯逻辑函数层 (数据解析与构建)
# =============================================================================
@app.route('/submission_details.html')
def submission_details():
    return render_template('submission_details.html')

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
        traceback.print_exc()

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
                            original_video_list: List[Dict], url_to_id_map: Dict[str, str]) -> Dict:
    """
    构建发布任务的存库数据结构
    [修复] 增加 url_to_id_map 参数，解决输入URL与库中URL不一致导致列表为空的问题
    """
    # 1. 获取本次任务所有有效的 video_id 集合
    valid_video_ids = set(m['video_id'] for m in materials)

    url_info_list = []

    # 2. 遍历用户输入的原始列表
    for item in original_video_list:
        input_url = item.get('original_url', '').strip()  # 记得 strip，保持一致

        # 从本次解析的 map 中获取 ID (这是最准确的对应关系)
        vid = url_to_id_map.get(input_url)

        # 3. 只有当这个 ID 存在于本次有效的 materials 中时，才加入列表
        if vid and vid in valid_video_ids:
            info_item = item.copy()
            info_item['video_id'] = vid
            url_info_list.append(info_item)

    return {
        'video_id_list': list(valid_video_ids),
        'userName': user_name,
        'status': TaskStatus.PROCESSING,
        'failed_count': 0,
        'original_url_info_list': url_info_list,  # 此时这里就不会为空了
        'creation_guidance_info': global_settings,
        'new_video_script_info': None,
        'upload_info': None,
        'create_time': datetime.now(),
    }


def check_cached_material(cached_material, video_item):
    """
    判断单个视频传入的信息是否改变。
    比较 db 中的 extra_info 和 传入的 video_item，
    但排除 '_cardDomId', 'original_url', 'is_realtime_video' 这几个字段。
    """
    # 1. 定义不需要比较的字段集合
    ignore_keys = {'_cardDomId', 'original_url', 'is_realtime_video', 'video_id'}

    # 2. 获取 DB 中的数据，处理可能为 None 的情况
    db_info = cached_material.get('extra_info') or {}

    # 3. 获取传入的数据，处理可能为 None 的情况
    current_info = video_item or {}

    # 4. 生成过滤后的字典（只包含未被忽略的字段）
    # 使用字典推导式：遍历原字典，只有 key 不在 ignore_keys 中才保留
    clean_current_info = {k: v for k, v in current_info.items() if k not in ignore_keys}

    # clean_db_info只保留clean_current_info中存在的key进行比较，避免db_info中有但current_info中没有的key影响结果
    clean_db_info = {k: v for k, v in db_info.items() if k in clean_current_info}

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
        all_path_info = build_video_paths(cached_material.get('video_id'))
        origin_video_path = all_path_info.get('origin_video_path')
        delete_files_in_dir_except_target(origin_video_path)
        cached_material['logical_scene_info'] = None
        cached_material['video_overlays_text_info'] = None
        cached_material['owner_asr_info'] = None
        cached_material['hudong_info'] = None
        cached_material['need_recut'] = True

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

def _fetch_and_map_db_materials(video_ids: List[str]) -> Dict[str, Dict]:
    """Helper: 根据 ID 列表批量查询数据库，返回 {video_id: material} 映射"""
    if not video_ids:
        return {}
    db_results = mongo_manager.find_materials_by_ids(video_ids)
    return {m['video_id']: m for m in db_results}

def _resolve_ids_and_fetch_missing_meta(input_video_list: List[Dict]) -> Tuple[Dict[str, str], Dict[str, Dict], Dict[str, Dict], List[str]]:
    """
    核心逻辑重构：
    Step 1: 遍历列表，获取 video_id。
            - 优先查本地文件缓存。
            - 本地没有则联网解析，并【缓存 meta_data】。
    Step 2: 拿着所有 video_id 查数据库。
    Step 3: 补全缺失的 meta_data。
            - 遍历所有 video_id，如果 DB 中没有，且 Step 1 中没解析过(即ID来自本地缓存)，
              则此时必须联网解析以获取 meta_data。

    Returns:
        url_to_id_map: URL -> video_id
        id_to_meta_map: video_id -> meta_data (只包含数据库中缺失且已解析到的)
        db_materials_map: video_id -> db_material
        errors: 错误信息
    """
    original_url_id_info = read_json(LOCAL_ORIGIN_URL_ID_INFO_PATH)
    is_url_mapping_updated = False

    url_to_id_map = {}
    id_to_meta_map = {} # 暂存解析到的 meta，用于后续构建，避免二次请求
    errors = []

    request_scope_parsed_cache = {} # 防止同一次请求中同一个URL重复解析

    # === Phase 1: 解析所有的 ID ===
    for idx, video_item in enumerate(input_video_list, start=1):
        url = video_item.get('original_url', '').strip()
        if not url:
            errors.append(f"第 {idx} 条记录错误: 视频链接为空")
            continue

        # 1.1 尝试从本地缓存获取
        local_video_id = original_url_id_info.get(url)

        if local_video_id:
            url_to_id_map[url] = local_video_id
            continue

        # 1.2 本地没有，必须网络解析
        if url in request_scope_parsed_cache:
            success, meta, err_msg = request_scope_parsed_cache[url]
        else:
            print(f"本地无缓存，执行解析: {url}")
            success, meta, err_msg = parse_douyin_video(url)
            request_scope_parsed_cache[url] = (success, meta, err_msg)

        if not success:
            errors.append(f"第 {idx} 条记录解析失败 (URL: {url}): {err_msg}")
            continue

        current_video_id = meta.get('id')
        if not current_video_id:
            errors.append(f"第 {idx} 条记录解析成功但无ID (URL: {url})")
            continue

        # 记录结果
        url_to_id_map[url] = current_video_id
        # 【关键点】：顺便保存 meta_data，防止后面发现 DB 没数据又要解析一次
        id_to_meta_map[current_video_id] = meta

        # 更新本地缓存
        original_url_id_info[url] = current_video_id
        is_url_mapping_updated = True

    # === Phase 2: 查询数据库 ===
    all_resolved_ids = list(url_to_id_map.values())
    # 去重
    all_resolved_ids = list(set(all_resolved_ids))
    db_materials_map = _fetch_and_map_db_materials(all_resolved_ids)

    # === Phase 3: 补全数据库缺失的 Meta Data ===
    # 此时，数据库没有的数据，我们需要确保 id_to_meta_map 里有。
    # Phase 1 中网络解析的已经有了，唯独缺的是：ID来自本地缓存，但 DB 被清空了的情况。

    # 建立 URL 到 ID 的反向查找或者直接遍历 input_video_list 对应的 URL
    # 为了效率，我们直接遍历 map
    for url, vid in url_to_id_map.items():
        # 如果数据库有，不需要 meta
        if vid in db_materials_map:
            continue

        # 如果数据库没有，检查 Phase 1 是否已经解析并存了 meta
        if vid in id_to_meta_map:
            continue

        # 走到这里说明：DB无数据，且 Phase 1 没解析（说明走的本地ID缓存）
        # 行动：现在解析
        print(f"数据缺失补全：ID {vid} 在本地缓存但不在数据库，重新解析元数据...")

        # 查缓存避免重复
        if url in request_scope_parsed_cache:
             success, meta, err_msg = request_scope_parsed_cache[url]
        else:
             success, meta, err_msg = parse_douyin_video(url)
             request_scope_parsed_cache[url] = (success, meta, err_msg)

        if success:
            id_to_meta_map[vid] = meta
        else:
            # 这里记录个错误，但可能前面 Phase 1 没报错，这里报错了比较尴尬
            # 不过一般来说 URL 之前能解析出 ID，现在大概率也能解析
            errors.append(f"补全元数据失败 (URL: {url}): {err_msg}")

    if is_url_mapping_updated:
        save_json(LOCAL_ORIGIN_URL_ID_INFO_PATH, original_url_id_info)

    return url_to_id_map, id_to_meta_map, db_materials_map, errors


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
            continue # 已在前面步骤报错，这里跳过

        video_id = url_to_id_map.get(url)
        if not video_id:
            continue # 解析失败的跳过

        # 批次内重复校验
        if video_id in current_batch_video_ids:
            errors.append(f"第 {idx} 条记录重复 (URL: {url}): ID {video_id} 已在当前任务列表中")
            continue

        # A. 检查数据库中是否存在 (优先)
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

        # B. 数据库无记录，使用 id_to_meta_map 中的数据构建
        elif video_id in id_to_meta_map:
            meta_data = id_to_meta_map[video_id]
            duration = meta_data.get('duration', 0)
            final_material = build_video_material_data(video_item, meta_data, video_id)

        else:
            # 理论上 Step 3 已经补全了所有情况。如果到这里还没数据，说明补全解析失败了。
            # 错误信息已经在 Step 3 或 Phase 1 加入了 errors 列表
            continue

        # C. 统一校验时间戳
        time_err, _ = validate_timestamp(video_item, duration)
        if time_err:
            errors.append(f"第 {idx} 条记录时间戳错误 (URL: {url}): {time_err}")
            continue

        final_material['video_overlays_text_info'] = None

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
# 3. 核心业务流程 (重构后 - 集成入队逻辑)
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

    # 2. 解析 ID 并确保数据完整性 (核心修改点)
    # 流程：解析ID(存Meta) -> 查库 -> 补全缺失Meta
    url_to_id_map, id_to_meta_map, db_materials_map, parse_errors = _resolve_ids_and_fetch_missing_meta(input_video_list)

    if parse_errors:
        response_structure['message'] = ErrorMessage.PARTIAL_PARSE_FAILURE
        response_structure['errors'] = parse_errors
        return response_structure, 400

    # 3. 构建与校验素材对象
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

    # 4. 任务查重
    if _check_task_duplication(user_name, valid_materials, global_settings):
        print(f"用户 {user_name} 提交的任务完全重复，跳过。")
        response_structure['status'] = ResponseStatus.SUCCESS
        response_structure['message'] = ErrorMessage.TASK_ALREADY_EXISTS
        response_structure['errors'] = ['可尝试采用不同的素材或者调整创作指导也能创建新任务']
        return response_structure, 500

    # 5. 保存数据并入队
    try:
        mongo_manager.upsert_materials(valid_materials)
        task_data = build_publish_task_data(user_name, global_settings, valid_materials, input_video_list, url_to_id_map)
        mongo_manager.upsert_tasks([task_data])

        # =========================================================
        # [修改] 成功保存后，将 video_id 入队并维护 running_task_ids
        # =========================================================
        if task_queue is not None:
            if check_task_queue(running_task_ids, task_data, check_time=False):
                # 加锁
                v_ids = task_data.get('video_id_list', [])
                for v_id in v_ids:
                    running_task_ids[v_id] = time.time()  # 确保写入当前时间
                task_queue.put(task_data)
                print(f"收到新{user_name} 入队成功个任务。当前时间 {time.strftime('%Y-%m-%d %H:%M:%S')} 队列大小: {task_queue.qsize()} {request_data}")
        else:
            print("⚠️ 警告: 任务队列未初始化，仅保存到数据库，未实时触发处理。")
        # =========================================================

        print(f"成功创建新任务，包含 {len(valid_materials)} 个视频。{task_data.get('userName')} {task_data.get('video_id_list')} ")
        response_structure['status'] = ResponseStatus.SUCCESS
        response_structure['message'] = f'新任务已成功创建，包含 {len(valid_materials)} 个视频。'
        return response_structure, 200

    except Exception as e:
        traceback.print_exc()

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

        # 增加日志输出：打印返回给前端的信息
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] One-Click Response: {response_data}")

        return jsonify(response_data), status_code

    except Exception as e:
        app.logger.exception("one_click_generate 接口发生未处理异常")
        error_response = {
            'status': ResponseStatus.ERROR,
            'message': '内部服务器错误',
            'errors': [str(e)]
        }
        # 增加日志输出：打印错误返回信息
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] One-Click Error Response: {error_response}")
        return jsonify(error_response), 500


@app.route('/check-video-status', methods=['POST'])
def check_video_status() -> Tuple[Response, int]:
    """检查视频状态接口"""
    try:
        data = request.get_json()
        response_data, status_code = process_check_video_status(data)

        # 增加日志输出：打印返回给前端的信息
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Check-Status Response: {response_data}")

        return jsonify(response_data), status_code
    except Exception as e:
        app.logger.exception("check_video_status 接口异常")
        error_response = {'status': 'error', 'message': str(e)}
        # 增加日志输出：打印错误返回信息
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Check-Status Error Response: {error_response}")
        return jsonify(error_response), 500


@app.route('/get_user_upload_info', methods=['GET'])
def get_user_upload_info() -> Response:
    try:
        user_name = request.args.get('userName', '').strip()
        user_upload_info = read_json(USER_STATISTIC_INFO_PATH)
        user_info = user_upload_info.get(user_name, {})
        response_data = {
            'status': ResponseStatus.SUCCESS,
            'message': '获取成功',
            'errors': [],
            'data': {
                'tomorrow_process': user_info.get('tomorrow_process', 0),
                'today_process': user_info.get('today_process', 0),
                'today_upload_count': user_info.get('today_upload_count', 0),
            }
        }
        # 增加日志输出：打印返回给前端的信息
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Upload-Info Response: {response_data}")

        return jsonify(response_data)
    except Exception as e:
        traceback.print_exc()
        app.logger.exception("get_user_upload_info 接口异常")
        error_response = {'status': 'error', 'message': str(e)}
        # 增加日志输出：打印错误返回信息
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Upload-Info Error Response: {error_response}")
        return jsonify(error_response)


def process_video_data(data: dict, video_type: str, user_name) -> dict:
    """
    根据输入的 video_type 处理数据，返回包含 tags, hot_videos, today_videos 的字典。
    """

    # --- 1. 处理 tags 字段 ---
    tags = []
    # 检查 good_tags_info 是否存在以及 video_type 是否在其中
    if "good_tags_info" in data and video_type in data["good_tags_info"]:
        tag_dict = data["good_tags_info"][video_type]
        sorted_tags = sorted(tag_dict.items(), key=lambda x: x[1], reverse=True)
        # 取排序后的前 5 个元素的 key (tag名字)
        tags = [item[0] for item in sorted_tags[:10]]

    # --- 2. 处理 hot_videos 字段 ---
    hot_videos = []
    if "good_video_list" in data:
        filtered_videos = [
            v for v in data["good_video_list"]
            if v.get("video_type") == video_type
        ]
        filtered_videos.sort(
            key=lambda x: len(x.get("choose_reason", [])),
            reverse=True
        )
        user_config = read_json(r'W:\project\python_project\auto_video\config\user_config.json')
        self_user_list = user_config.get('self_user_list', [])

        # 过滤掉final_good_task_list中userName在self_user_list中的任务
        filtered_videos = [task_info for task_info in filtered_videos if task_info.get('userName', '') not in self_user_list]
        filtered_videos = [task_info for task_info in filtered_videos if task_info.get('final_score', 0) > 100]

        # 第三步：取前 5 个，并提取 title 和 bvid
        for video in filtered_videos[:5]:
            title = ""
            # 尝试获取标题：优先从 upload_params 获取（通常是最终标题）
            if "upload_params" in video and "title" in video["upload_params"]:
                title = video["upload_params"]["title"]
            # 备选：从 upload_info 列表的第一个元素获取
            elif "upload_info" in video and isinstance(video["upload_info"], list) and len(video["upload_info"]) > 0:
                title = video["upload_info"][0].get("title", "")

            hot_videos.append({
                "title": title,
                "url": f"https://www.bilibili.com/video/{video.get("bvid", "")}"
            })

    # --- 3. 处理 today_videos 字段 ---
    today_videos = build_today_videos(user_name)

    # --- 返回结果 ---
    return {
        "tags": tags,
        "hot_videos": hot_videos,
        "today_videos": today_videos
    }

def build_today_videos(user_name):
    """
    构建今日视频数据
    :param user_name:
    :return:
    """
    current_time = datetime.now()
    # 计算一天前的时间
    one_day_ago = current_time - timedelta(days=1)

    query_2 = {
        "userName": user_name,
        "create_time": {
            "$gt": one_day_ago
        }
    }

    all_task = mongo_manager.find_by_custom_query(mongo_manager.tasks_collection, query_2)

    today_videos = []
    for task_info in all_task:
        temp_dict = {}
        creation_guidance_info = task_info.get('creation_guidance_info', {})
        creative_guidance = creation_guidance_info.get('creative_guidance', {})
        if not creative_guidance:
            continue
        temp_dict['creative_guidance'] = creative_guidance
        create_time = task_info.get('create_time')
        # 将create_time转换成为字符串，不需要年的信息
        create_time_str = create_time.strftime("%m-%d %H:%M")
        temp_dict['created_at'] = create_time_str
        original_url_info_list = task_info.get('original_url_info_list', [])
        original_url_list = [info.get('original_url') for info in original_url_info_list]
        temp_dict['origin_url_list'] = original_url_list
        upload_detail = '处理中'
        failed_count = task_info.get('failed_count', 0)
        if task_info.get('status') in [TaskStatus.TO_UPLOADED, TaskStatus.PLAN_GENERATED]:
            upload_detail = '处理中'
        elif task_info.get('status') == TaskStatus.FAILED and failed_count > VIDEO_MAX_RETRY_TIMES:
            upload_detail = f'失败_{task_info.get('failure_details', '')}'
        bvid = task_info.get('bvid', '')
        if bvid:
            upload_detail = f"https://www.bilibili.com/video/{bvid}"
        temp_dict['upload_detail'] = upload_detail
        upload_params = task_info.get('upload_params', {})
        title = upload_params.get('title', '')
        temp_dict['title'] = title
        today_videos.append(temp_dict)
    return today_videos


@app.route('/get_good_video', methods=['GET'])
def get_good_video_info():
    user_name = request.args.get('username')
    print(f"接收到的用户名: {user_name}")
    user_type = get_user_type(user_name)
    statistic_play_info = read_json(STATISTIC_PLAY_COUNT_FILE)
    data_info = process_video_data(statistic_play_info, user_type, user_name)
    print(f"处理后的视频数据: {data_info}")
    return jsonify(data_info)


# =============================================================================
# 5. 进程监控与主程序入口
# =============================================================================

def _monitor_processes():
    """
    [新增] 后台监控线程：专门用于监控和重启挂掉的子进程。
    必须放在独立线程中，否则会阻塞 Flask 的运行。
    """
    global producer_p, consumers, task_queue, running_task_ids
    print("👀 进程监控线程已启动...")

    while True:
        try:
            # 1. 监控消费者
            for i in range(len(consumers)):
                p = consumers[i]
                if not p.is_alive():
                    print(f"警告: 消费者进程 {p.pid} 挂了，重启中...")
                    new_p = multiprocessing.Process(
                        target=_task_process_worker,
                        args=(task_queue, running_task_ids)
                    )
                    new_p.daemon = True
                    new_p.start()
                    consumers[i] = new_p

            # 2. 监控生产者
            if producer_p and not producer_p.is_alive():
                print(f"严重警告: 生产者进程 {producer_p.pid} 挂了，立即重启！")
                producer_p = multiprocessing.Process(
                    target=_task_producer_worker,
                    args=(task_queue, running_task_ids)
                )
                producer_p.daemon = True
                producer_p.start()

            time.sleep(60)  # 每分钟检查一次
        except Exception as e:
            print(f"监控线程发生错误: {e}")
            time.sleep(60)

if __name__ == "__main__":
    # 1. 初始化 Multiprocessing Manager
    global_manager = multiprocessing.Manager()

    # 2. 初始化共享对象
    # 共享去重字典 (Key: video_id)
    running_task_ids = global_manager.dict()
    # 任务队列
    task_queue = multiprocessing.Queue()

    # 3. 启动消费者集群
    max_workers = 10
    print(f"主线程: 启动 {max_workers} 个消费者进程...")

    for _ in range(max_workers):
        p = multiprocessing.Process(
            target=_task_process_worker,
            args=(task_queue, running_task_ids)
        )
        p.daemon = True
        p.start()
        consumers.append(p)

    # 4. 启动生产者进程
    print(f"主线程: 启动 1 个生产者进程...")
    producer_p = multiprocessing.Process(
        target=_task_producer_worker,
        args=(task_queue, running_task_ids)
    )
    producer_p.daemon = True
    producer_p.start()

    # 5. 启动后台监控线程 (关键：不能阻塞主线程，因为主线程要运行 Flask)
    monitor_thread = threading.Thread(target=_monitor_processes, daemon=True)
    monitor_thread.start()

    # 6. 启动 Flask
    print("Flask 接口服务启动...")
    app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=False)
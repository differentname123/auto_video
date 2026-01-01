"""
视频处理与发布任务管理服务

该模块提供Flask Web服务，用于接收前端提交的视频处理任务，
解析抖音视频元数据，并将任务信息存储到MongoDB中。
重构版：函数式编程风格，扁平化结构，逻辑解耦。
包含优化：
1. 优先利用本地映射和数据库缓存，减少重复解析。
2. 任务查重加入 creative_guidance 校验，支持同素材不同指令的多次生成。
3. 确保所有涉及的素材（无论新旧）都会再次执行保存操作。
"""

import time
from typing import Optional, List, Tuple, Dict, Any

from flask import Flask, request, jsonify, render_template, Response

from utils.common_utils import read_json, save_json
# 导入配置和工具
from video_common_config import TaskStatus, _configure_third_party_paths, ErrorMessage, ResponseStatus, \
    ALLOWED_USER_LIST, LOCAL_ORIGIN_URL_ID_INFO_PATH

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


def build_video_material_data(video_item: Dict, meta_data: Dict, video_id: str) -> Dict:
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
    # 注意：materials可能来自数据库，也可能来自新解析，需确保base_info中有original_url
    successful_urls = {}
    for m in materials:
        # 兼容处理：如果是新解析的在base_info里，如果是数据库读出来的也在base_info里
        original_url = m.get('base_info', {}).get('original_url')
        if original_url:
            successful_urls[original_url] = m['video_id']

    url_info_list = []

    for item in original_video_list:
        url = item.get('original_url')
        if url in successful_urls:
            # 复制一份 item 避免修改原始数据
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
    步骤：校验 -> 尝试获取缓存/批量解析 -> 智能查重 -> 保存素材 -> 保存任务
    """
    # --- Step 1: 基础参数校验 ---
    if not request_data or not request_data.get('userName') or not request_data.get('video_list'):
        return {'status': ResponseStatus.ERROR, 'message': ErrorMessage.MISSING_REQUIRED_FIELDS}, 400

    user_name = request_data['userName']
    if user_name not in ALLOWED_USER_LIST:
        return {'status': ResponseStatus.ERROR, 'message': f"用户名{user_name}未向管理员注册"}, 403

    global_settings = request_data.get('global_settings', {})
    input_video_list = request_data['video_list']

    print(f"开始处理请求 | 用户: {user_name} | 视频数: {len(input_video_list)}")

    # --- Step 2: 获取视频素材 (优先查本地缓存+DB，失败则解析) ---
    valid_materials = []
    errors = []
    original_url_id_info = read_json(LOCAL_ORIGIN_URL_ID_INFO_PATH)
    is_url_mapping_updated = False  # 标记是否需要回写本地json

    for idx, video_item in enumerate(input_video_list, start=1):
        url = video_item.get('original_url', '').strip()
        if not url:
            errors.append(f"第 {idx} 条视频链接为空")
            continue

        # [需求1优化]：先尝试从本地映射ID + 数据库内容获取
        cached_material = None
        local_video_id = original_url_id_info.get(url)

        if local_video_id:
            # 如果本地有ID，尝试去数据库查完整数据
            # find_materials_by_ids 通常返回列表，我们取第一个
            db_results = mongo_manager.find_materials_by_ids([local_video_id])
            if db_results and len(db_results) > 0:
                print(f"URL命中缓存，跳过解析: {url}")
                # 使用数据库中的数据，并更新extra_info以匹配当前请求上下文（如果需要）
                cached_material = db_results[0]
                # 确保 extra_info 是当前请求传入的（可能包含新的时间戳或其他标记）
                cached_material['extra_info'] = video_item

        if cached_material:
            valid_materials.append(cached_material)
        else:
            # 缓存未命中（无ID 或 有ID但库里没数据），执行解析
            print(f"缓存未命中，开始解析: {url}")
            success, meta, err_msg = parse_douyin_video(url)

            if not success:
                errors.append(f"第 {idx} 条解析失败: {err_msg}")
                continue

            video_id = meta.get('id')
            if not video_id:
                errors.append(f"第 {idx} 条解析成功但无ID")
                continue

            # 更新本地映射
            original_url_id_info[url] = video_id
            is_url_mapping_updated = True

            # 调用纯函数构建素材数据
            material = build_video_material_data(video_item, meta, video_id)
            valid_materials.append(material)

    # 仅当有更新时回写本地文件
    if is_url_mapping_updated:
        save_json(LOCAL_ORIGIN_URL_ID_INFO_PATH, original_url_id_info)

    if errors:
        return {
            'status': ResponseStatus.ERROR,
            'message': ErrorMessage.PARTIAL_PARSE_FAILURE,
            'errors': errors
        }, 400

    if not valid_materials:
        return {'status': ResponseStatus.ERROR, 'message': "无有效视频可处理"}, 400

    # --- Step 3: 任务查重 (优化版：同时校验 creative_guidance) ---
    video_ids = [m['video_id'] for m in valid_materials]

    # 获取数据库中包含这些 video_ids 的已有任务
    # 假设 find_task_by_exact_video_ids 返回的是匹配的任务列表(或单个任务对象)，而非简单的True/False
    # 如果原方法只返回True，需要修改 mongo_manager 让其返回具体数据。
    # 这里假定它返回所有 video_id_list 完全匹配的任务列表。
    existing_tasks = mongo_manager.find_task_by_exact_video_ids(video_ids)

    # [需求2优化]：判定重复逻辑
    is_duplicate_task = False
    current_guidance = global_settings.get('creative_guidance', '')

    if existing_tasks:
        # 统一转为列表处理
        if not isinstance(existing_tasks, list):
            existing_tasks = [existing_tasks]

        for task in existing_tasks:
            # 获取已有任务的 creative_guidance
            task_guidance_info = task.get('creation_guidance_info', {})
            # 兼容有些历史数据可能没有 creation_guidance_info 字段的情况
            if task_guidance_info is None:
                task_guidance_info = {}

            old_guidance = task_guidance_info.get('creative_guidance', '')

            # 如果 video_ids 相同 且 guidance 也相同，才判定为重复
            if old_guidance == current_guidance:
                is_duplicate_task = True
                break

    if is_duplicate_task:
        print(f"用户 {user_name} 提交的任务完全重复（视频+Prompt相同），跳过。Ids: {video_ids}")
        return {'status': ResponseStatus.SUCCESS, 'message': ErrorMessage.TASK_ALREADY_EXISTS}, 200

    # --- Step 4: 保存视频素材 (独立步骤) ---
    # [需求3优化]：无论素材是查出来的还是新解析的，都执行 upsert 操作
    try:
        mongo_manager.upsert_materials(valid_materials)
        print(f"成功保存/更新 {len(valid_materials)} 条视频素材。")
    except Exception as e:
        app.logger.error(f"保存素材失败: {e}")
        return {'status': ResponseStatus.ERROR, 'message': "数据库错误: 无法保存视频素材"}, 500

    # --- Step 5: 构建并保存任务 (独立步骤) ---
    try:
        # 调用纯函数构建任务数据
        task_data = build_publish_task_data(user_name, global_settings, valid_materials, input_video_list)
        mongo_manager.upsert_tasks([task_data])
        print(f"成功创建新任务，包含 {len(valid_materials)} 个视频。")
    except Exception as e:
        app.logger.error(f"保存任务失败: {e}")
        return {'status': ResponseStatus.ERROR, 'message': "数据库错误: 无法创建任务"}, 500

    return {'status': ResponseStatus.SUCCESS, 'message': f'新任务已成功创建，包含 {len(valid_materials)} 个视频。'}, 200


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
            'message': f'内部服务器错误: {str(e)}'
        }), 500


@app.route('/get_user_upload_info', methods=['GET'])
def get_user_upload_info() -> Response:
    """获取用户上传统计信息 (Mock)"""
    return jsonify({
        'status': ResponseStatus.SUCCESS,
        'data': {
            'total_count_today': 0,
            'unprocessed_count_today': 0,
            'remote_upload_count': 0
        }
    })


if __name__ == "__main__":
    print("Flask 接口服务启动...")
    app.run(host='0.0.0.0', port=5002, debug=True, use_reloader=False)
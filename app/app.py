import sys
import os

from utils.mongo_base import gen_db_object
from utils.mongo_manager import MongoManager

# --- 新的、修正后的代码 ---
# 1. 先计算出项目的根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 2. 构造出那个行为不标准的第三方库的根目录
#    它的内部代码（如 console.py）是基于这个目录来写 'from src...' 的
downloader_root = os.path.join(project_root, 'third_party', 'TikTokDownloader')

# 3. 将这个特定的库目录添加到 sys.path 中
#    这样 'from src' 就能在这个目录里找到 src 文件夹了
if downloader_root not in sys.path:
    sys.path.insert(0, downloader_root)
# --- 代码修改结束 ---



from flask import Flask, request, jsonify, render_template

from third_party.TikTokDownloader.douyin_downloader import get_meta_info

app = Flask(__name__)

print("Initializing MongoDB connection for the application...")
mongo_base_instance = gen_db_object()
manager = MongoManager(mongo_base_instance)
print("✅ MongoDB Manager is ready.")
def parse_douyin_url(video_url):
    """
    解析出抖音视频信息
    """
    try:
        meta_data_list = get_meta_info(video_url)
        if not meta_data_list:  # 检查列表是否为空
            return "解析失败：未能从链接中提取到任何元数据", None

        # 假设总是取第一个结果
        return "", meta_data_list[0]
    except Exception as e:
        print(f"解析URL '{video_url}' 时发生异常: {e}")
        return f"解析异常: {e}", None


# --- 辅助函数：用于创建 video_materials 数据 ---
def _create_video_material(video_item, global_settings, meta_data, video_id):
    """
    根据输入信息，组装单个 video_materials 对象。

    Args:
        video_item (dict): 来自前端 video_list 的单个视频项。
        global_settings (dict): 来自前端的全局设置。
        meta_data (dict): 解析视频URL后得到的元数据。
        video_id (str): 为该视频生成的唯一ID。

    Returns:
        dict: 符合 video_materials 表结构的字典。
    """
    return {
        'video_id': video_id,
        'status': '处理中',
        'error_info': None,
        'base_info': {
            'video_title': meta_data.get('full_title') or meta_data.get('desc'),
            'video_desc': meta_data.get('desc'),
            'collection_time': meta_data.get('collection_time'),
            'author': meta_data.get('nickname'),
            'upload_time': meta_data.get('create_time'),
            'duration': meta_data.get('duration'),
            'tags': meta_data.get('text_extra', []),
            'height': meta_data.get('height'),
            'width': meta_data.get('width'),
            'original_url': video_item.get('url'),
            'download_url': meta_data.get('downloads'),  # 注意：这是临时链接，后续应更新为云存储地址
            'dynamic_cover': meta_data.get('dynamic_cover'),
            'static_cover': meta_data.get('static_cover'),
            'comment_list': [],  # 缺失字段，需要后续获取
            'digg_count': meta_data.get('digg_count'),
            'comment_count': meta_data.get('comment_count'),
            'collect_count': meta_data.get('collect_count'),
            'share_count': meta_data.get('share_count')
        },
        'extra_info': {
            'is_requires_text': video_item.get('need_text'),
            'is_needs_stickers': video_item.get('need_emoji'),
            'is_needs_audio_replace': global_settings.get('audio_replace'),
            'is_realtime_video': video_item.get('is_realtime'),
            # 以下字段需要后续处理才能填充
            'scene_timestamp_list': None,
            'logical_scene_info': None,
            'asr_info': None,
            'owner_subtitle_box_info': None
        }
    }


# --- 辅助函数：用于创建 publish_tasks 数据 ---
def _create_publish_task(user_name, global_settings, video_materials_list):
    """
    根据本次请求的所有信息，组装 publish_tasks 对象。

    Args:
        user_name (str): 用户名。
        global_settings (dict): 全局设置。
        video_materials_list (list): 本次任务生成的所有 video_materials 对象列表。

    Returns:
        dict: 符合 publish_tasks 表结构的字典。
    """
    # 从 video_materials_list 中提取任务所需信息
    video_id_list = [m['video_id'] for m in video_materials_list]
    original_url_info_list = [
        {
            'origin_url': m['base_info']['original_url'],
            'video_id': m['video_id'],
            # 注意: 'is_contains_creator_voice' 不在 material 中, 需要从原始 video_item 获取
            # 这里为了简化，我们假设它可以通过其他方式回溯，或在主流程中构建
            'is_requires_text': m['extra_info']['is_requires_text'],
            'is_needs_stickers': m['extra_info']['is_needs_stickers'],
            'is_needs_audio_replace': m['extra_info']['is_needs_audio_replace'],
            'is_realtime_video': m['extra_info']['is_realtime_video']
        }
        for m in video_materials_list
    ]

    return {
        'video_id_list': video_id_list,
        'userName': user_name,
        'original_url_info_list': original_url_info_list,  # 注意：此实现缺少 is_contains_creator_voice
        'creation_guidance_info': {
            'retain_ratio': global_settings.get('retention_ratio'),
            'is_need_original': global_settings.get('is_original'),
            'is_need_narration': global_settings.get('allow_commentary'),
            'is_need_scene_title_and_summary': global_settings.get('scene_title')
        },
        'new_video_script_info': None,  # 缺失字段，后续生成
        'expected_publish_time': global_settings.get('schedule_date'),
        'upload_info': None  # 缺失字段，后续生成
    }

@app.route('/one-click-generate', methods=['POST'])
def one_click_generate():
    """
    接收前端提交的任务请求，并提供准确的响应信息。
    """
    try:
        # --- 阶段 0: 数据准备 (逻辑不变) ---
        data = request.get_json()
        if not data:
            return jsonify({'status': 'error', 'message': '请求体为空'}), 400

        user_name = data.get('userName')
        global_settings = data.get('global_settings', {})
        video_list = data.get('video_list', [])

        if not user_name or not video_list:
            return jsonify({'status': 'error', 'message': '用户名或视频列表为空'}), 400

        print(f"收到请求 | 用户: {user_name} | 视频数: {len(video_list)}")

        # --- 阶段 1: 解析和验证 ---
        generated_materials = []
        parsing_errors = []

        for index, video_item in enumerate(video_list):
            video_url = video_item.get('url', '').strip()
            if not video_url:
                # 可以选择记录一个错误或直接跳过
                parsing_errors.append(f"第 {index + 1} 条视频链接为空。")
                continue

            error_info, meta_data = parse_douyin_url(video_url)

            if error_info or not meta_data:
                # 需求1：收集详细的错误信息
                short_url = video_url[-20:] # 取URL后20位，避免太长
                error_msg = f"第 {index + 1} 条视频链接 (...{short_url}) 解析失败，原因：{error_info or '未知'}"
                parsing_errors.append(error_msg)
                print(error_msg)
                continue # 继续解析下一个，而不是中断

            video_id = meta_data.get('id')
            if not video_id:
                short_url = video_url[-20:]
                error_msg = f"第 {index + 1} 条视频链接 (...{short_url}) 解析成功但缺少'id'字段"
                parsing_errors.append(error_msg)
                continue

            material = _create_video_material(video_item, global_settings, meta_data, video_id)
            generated_materials.append(material)

        # 如果在解析阶段就有错误，则立即返回，不进行后续操作
        if parsing_errors:
            return jsonify({
                'status': 'error',
                'message': '部分视频解析失败，任务未创建。',
                'errors': parsing_errors # 将所有错误信息返回给前端
            }), 400

        # --- 阶段 2: 任务存在性检查 ---
        video_id_list_for_task = [m['video_id'] for m in generated_materials]

        # 需求2：查询任务是否已存在
        # 注意：find_task_by_exact_video_ids 内部会自动排序
        existing_task = manager.find_task_by_exact_video_ids(video_id_list_for_task)
        if existing_task:
            print("检测到重复任务，已存在相同的视频ID列表，跳过创建新任务。")
            return jsonify({
                'status': 'success', # 操作是成功的，只是结果是“已存在”
                'message': '任务已存在，无需重复创建。'
            })

        # --- 阶段 3: 数据写入 (只有在解析全部成功且任务不存在时执行) ---

        # 3.1 批量写入素材
        if generated_materials:
            manager.upsert_materials(generated_materials)

        # 3.2 构造并写入任务
        original_url_info_list = []
        successful_urls = {m['base_info']['original_url'] for m in generated_materials}
        for item in video_list:
             if item.get('url') in successful_urls:
                mat = next((m for m in generated_materials if m['base_info']['original_url'] == item['url']), None)
                if mat:
                     original_url_info_list.append({
                        'origin_url': item['url'],
                        'video_id': mat['video_id'],
                        'is_contains_creator_voice': item.get('has_author_voice'),
                        'is_requires_text': item.get('need_text'),
                        # ... 其他字段 ...
                    })

        publish_task_data = {
            'video_id_list': video_id_list_for_task,
            'userName': user_name,
            'original_url_info_list': original_url_info_list,
            'creation_guidance_info': global_settings, # 可以简化
            'new_video_script_info': None,
            'expected_publish_time': global_settings.get('schedule_date'),
            'upload_info': None
        }
        manager.upsert_tasks([publish_task_data])

        return jsonify({
            'status': 'success',
            'message': f'新任务已成功创建，包含 {len(generated_materials)} 个视频。'
        })

    except Exception as e:
        app.logger.exception("An unhandled exception occurred in one_click_generate")
        return jsonify({'status': 'error', 'message': f'内部服务器错误: {e}'}), 500


# ==============================================================================
# 2. 辅助统计接口 (Mock)
# ==============================================================================

@app.route('/get_user_upload_info', methods=['GET'])
def get_user_upload_info():
    """
    前端页面加载时调用的统计接口
    """
    user_name = request.args.get('userName')

    # 这里直接返回模拟数据，让前端页面能正常显示
    return jsonify({
        'status': 'success',
        'data': {
            'total_count_today': 0,  # 今日投稿
            'unprocessed_count_today': 0,  # 待处理
            'remote_upload_count': 0  # 已上传
        }
    })


@app.route('/')
def index():
    return render_template('index.html')


# ==============================================================================
# 3. 启动
# ==============================================================================

if __name__ == "__main__":
    # threaded=True 允许 Flask 并行处理多个请求 (默认就是 True)
    print("Flask 接口服务启动...")
    app.run(host='0.0.0.0', port=5002, debug=True, use_reloader=False)

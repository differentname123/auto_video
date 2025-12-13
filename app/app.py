import sys
import os

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


def parse_douyin_url(video_url):
    """
    解析出抖音视频信息
    :param video_url:
    :return:
    """
    meta_data = get_meta_info(video_url)
    error_info = ""
    if not meta_data:
        error_info = "未能从链接中提取到有效的视"

    return error_info, meta_data[0]


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
    接收前端提交的任务请求
    只负责：接收数据 -> 解析合并参数 -> 生成数据结构 -> 立即返回
    """
    try:
        # 1. 获取前端 JSON 数据
        data = request.get_json()
        if not data:
            return jsonify({'status': 'error', 'message': '请求体为空'}), 400

        # 2. 提取顶层参数
        user_name = data.get('userName')
        global_settings = data.get('global_settings', {})
        video_list = data.get('video_list', [])

        # 简单的权限/非空校验
        if not user_name:
            return jsonify({'status': 'error', 'message': '未提供用户名'}), 400
        if not video_list:
            return jsonify({'status': 'error', 'message': '视频列表为空'}), 400

        print(f"收到请求 | 用户: {user_name} | 视频数: {len(video_list)}")

        # 3. 遍历视频列表，处理每个视频
        generated_materials = []
        for video_item in video_list:
            video_url = video_item.get('url')
            if not video_url:
                continue  # 跳过没有URL的项

            # 调用你的真实URL解析函数
            error_info, meta_data = parse_douyin_url(video_url)

            if error_info or not meta_data:
                print(f"解析失败: {video_url} | 原因: {error_info}")
                # 你可以根据业务需求选择是中断还是跳过
                continue

            # 生成唯一ID并创建 video_material 对象
            video_id = meta_data['id']
            material = _create_video_material(video_item, global_settings, meta_data, video_id)
            generated_materials.append(material)

        # 4. 基于生成的所有 material，创建 publish_task 对象
        # 注意: _create_publish_task 的简化实现中缺少 is_contains_creator_voice
        # 更健壮的实现是在主流程中构建 original_url_info_list
        # 我们在这里直接构建以保证数据完整性

        video_id_list_for_task = [m['video_id'] for m in generated_materials]
        original_url_info_list = []
        for i, material in enumerate(generated_materials):
            original_url_info_list.append({
                'origin_url': material['base_info']['original_url'],
                'video_id': material['video_id'],
                'is_contains_creator_voice': video_list[i].get('has_author_voice'),
                'is_requires_text': material['extra_info']['is_requires_text'],
                'is_needs_stickers': material['extra_info']['is_needs_stickers'],
                'is_needs_audio_replace': material['extra_info']['is_needs_audio_replace'],
                'is_realtime_video': material['extra_info']['is_realtime_video'],
            })

        publish_task_data = {
            'video_id_list': video_id_list_for_task,
            'userName': user_name,
            'original_url_info_list': original_url_info_list,
            'creation_guidance_info': {
                'retain_ratio': global_settings.get('retention_ratio'),
                'is_need_original': global_settings.get('is_original'),
                'is_need_narration': global_settings.get('allow_commentary'),
                'is_need_scene_title_and_summary': global_settings.get('scene_title')
            },
            'new_video_script_info': None,
            'expected_publish_time': global_settings.get('schedule_date'),
            'upload_info': None
        }


        # 6. 立即返回成功，实现“快速接收”
        return jsonify({
            'status': 'success',
            'message': f'服务端已接收 {len(generated_materials)} 个任务'
        })

    except Exception as e:
        # 在生产环境中，建议使用日志库记录异常
        print(f"接口异常: {e}")
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

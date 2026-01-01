"""
视频处理与发布任务管理服务

该模块提供Flask Web服务，用于接收前端提交的视频处理任务，
解析抖音视频元数据，并将任务信息存储到MongoDB中。
"""

import time
from typing import Optional
from dataclasses import dataclass

from flask import Flask, request, jsonify, render_template, Response

from video_common_config import TaskStatus, _configure_third_party_paths, ErrorMessage, ResponseStatus


_configure_third_party_paths()


from utils.mongo_base import gen_db_object
from utils.mongo_manager import MongoManager
from third_party.TikTokDownloader.douyin_downloader import get_meta_info


@dataclass
class ParseResult:
    """视频URL解析结果"""
    success: bool
    meta_data: Optional[dict] = None
    error_message: Optional[str] = None



app = Flask(__name__)


def _init_mongo_manager() -> MongoManager:
    """初始化MongoDB管理器"""
    print("Initializing MongoDB connection for the application...")
    mongo_base_instance = gen_db_object()
    manager = MongoManager(mongo_base_instance)
    print("✅ MongoDB Manager is ready.")
    return manager


mongo_manager = _init_mongo_manager()


# =============================================================================
# 视频解析服务
# =============================================================================

class DouyinVideoParser:
    """抖音视频解析器"""

    @staticmethod
    def parse(video_url: str) -> ParseResult:
        """
        解析抖音视频URL，提取元数据。

        Args:
            video_url: 抖音视频链接

        Returns:
            ParseResult: 包含解析状态和元数据的结果对象
        """
        try:
            meta_data_list = get_meta_info(video_url)

            if not meta_data_list:
                return ParseResult(
                    success=False,
                    error_message=ErrorMessage.PARSE_NO_METADATA
                )

            return ParseResult(success=True, meta_data=meta_data_list[0])

        except Exception as e:
            print(f"解析URL '{video_url}' 时发生异常: {e}")
            return ParseResult(success=False, error_message=f"解析异常: {e}")


# =============================================================================
# 数据构建器
# =============================================================================

class VideoMaterialBuilder:
    """视频素材数据构建器"""

    @staticmethod
    def build(
            video_item: dict,
            meta_data: dict,
            video_id: str
    ) -> dict:
        """
        构建视频素材数据对象。

        Args:
            video_item: 前端提交的单个视频项
            global_settings: 全局设置
            meta_data: 视频元数据
            video_id: 视频唯一标识

        Returns:
            符合 video_materials 表结构的字典
        """
        return {
            'video_id': video_id,
            'status': TaskStatus.PROCESSING,
            'error_info': None,
            'base_info': VideoMaterialBuilder._build_base_info(video_item, meta_data),
            'extra_info': video_item
        }

    @staticmethod
    def _build_base_info(video_item: dict, meta_data: dict) -> dict:
        """构建视频基础信息"""
        return {
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
            'comment_list': [],
            'digg_count': meta_data.get('digg_count'),
            'comment_count': meta_data.get('comment_count'),
            'collect_count': meta_data.get('collect_count'),
            'share_count': meta_data.get('share_count')
        }



class PublishTaskBuilder:
    """发布任务数据构建器"""

    @staticmethod
    def build(
            user_name: str,
            global_settings: dict,
            video_materials: list,
            video_list: list
    ) -> dict:
        """
        构建发布任务数据对象。

        Args:
            user_name: 用户名
            global_settings: 全局设置
            video_materials: 视频素材列表
            video_list: 原始视频列表（来自前端）

        Returns:
            符合 publish_tasks 表结构的字典
        """
        video_id_list = [m['video_id'] for m in video_materials]
        original_url_info_list = PublishTaskBuilder._build_url_info_list(
            video_materials, video_list
        )

        return {
            'video_id_list': video_id_list,
            'userName': user_name,
            'status': TaskStatus.PROCESSING,
            'failed_count': 0,
            'original_url_info_list': original_url_info_list,
            'creation_guidance_info': global_settings,
            'new_video_script_info': None,
            'upload_info': None
        }

    @staticmethod
    def _build_url_info_list(video_materials: list, video_list: list) -> list:
        """构建原始URL信息列表"""
        successful_urls = {m['base_info']['original_url'] for m in video_materials}
        url_info_list = []

        for item in video_list:
            item_url = item.get('original_url')
            if item_url not in successful_urls:
                continue

            material = next(
                (m for m in video_materials if m['base_info']['original_url'] == item_url),
                None
            )
            if not material:
                continue
            item['video_id'] = material['video_id']
            url_info_list.append(item)

        return url_info_list


# =============================================================================
# 请求处理服务
# =============================================================================

class OneClickGenerateService:
    """一键生成服务"""

    def __init__(self, manager: MongoManager):
        self.manager = manager
        self.parser = DouyinVideoParser()

    def process(self, data: dict) -> tuple[dict, int]:
        """
        处理一键生成请求。

        Args:
            data: 请求数据

        Returns:
            (响应数据, HTTP状态码)
        """
        # 参数验证
        validation_error = self._validate_request(data)
        if validation_error:
            return validation_error

        user_name = data['userName']
        global_settings = data.get('global_settings', {})
        video_list = data['video_list']

        print(f"收到请求 | 用户: {user_name} | 视频数: {len(video_list)}")

        # 解析视频
        materials, errors = self._parse_videos(video_list)

        if errors:
            print(f"解析过程中出现错误: {errors}")
            return self._error_response(
                ErrorMessage.PARTIAL_PARSE_FAILURE,
                errors=errors
            ), 400

        # 检查任务是否已存在
        video_ids = [m['video_id'] for m in materials]
        if self.manager.find_task_by_exact_video_ids(video_ids):
            print(f"{video_ids} {user_name} 检测到重复任务，跳过创建。")
            return self._success_response(ErrorMessage.TASK_ALREADY_EXISTS), 200

        # 保存数据
        self._save_task(user_name, global_settings, materials, video_list)

        return self._success_response(
            f'新任务已成功创建，包含 {len(materials)} 个视频。'
        ), 200

    def _validate_request(self, data: dict) -> Optional[tuple[dict, int]]:
        """验证请求数据"""
        if not data:
            return self._error_response(ErrorMessage.EMPTY_REQUEST_BODY), 400

        if not data.get('userName') or not data.get('video_list'):
            return self._error_response(ErrorMessage.MISSING_REQUIRED_FIELDS), 400

        return None

    def _parse_videos(
            self,
            video_list: list,
    ) -> tuple[list, list]:
        """
        批量解析视频。

        Returns:
            (素材列表, 错误列表)
        """
        materials = []
        errors = []

        for index, video_item in enumerate(video_list, start=1):
            video_url = video_item.get('original_url', '').strip()

            if not video_url:
                errors.append(f"第 {index} 条视频链接为空。")
                continue

            result = self.parser.parse(video_url)

            if not result.success:
                url_suffix = video_url[-20:]
                errors.append(
                    f"第 {index} 条视频链接 (...{url_suffix}) 解析失败，"
                    f"原因：{result.error_message or '未知'}"
                )
                continue

            video_id = result.meta_data.get('id')
            if not video_id:
                url_suffix = video_url[-20:]
                errors.append(
                    f"第 {index} 条视频链接 (...{url_suffix}) 解析成功但缺少'id'字段"
                )
                continue

            material = VideoMaterialBuilder.build(
                video_item, result.meta_data, video_id
            )
            materials.append(material)

        return materials, errors

    def _save_task(
            self,
            user_name: str,
            global_settings: dict,
            materials: list,
            video_list: list
    ) -> None:
        """保存任务和素材到数据库"""
        if materials:
            self.manager.upsert_materials(materials)

        task_data = PublishTaskBuilder.build(
            user_name, global_settings, materials, video_list
        )
        self.manager.upsert_tasks([task_data])

    @staticmethod
    def _success_response(message: str) -> dict:
        """构建成功响应"""
        return {'status': ResponseStatus.SUCCESS, 'message': message}

    @staticmethod
    def _error_response(message: str, errors: list = None) -> dict:
        """构建错误响应"""
        response = {'status': ResponseStatus.ERROR, 'message': message}
        if errors:
            response['errors'] = errors
        return response


# 初始化服务实例
one_click_service = OneClickGenerateService(mongo_manager)


# =============================================================================
# API路由
# =============================================================================

@app.route('/')
def index() -> str:
    """首页"""
    return render_template('index.html')


@app.route('/one-click-generate', methods=['POST'])
def one_click_generate() -> tuple[Response, int]:
    """
    一键生成接口。

    接收前端提交的视频任务请求，解析视频元数据，
    创建处理任务并存储到数据库。
    """
    try:
        data = request.get_json()
        print(f"当前时间 {time.strftime('%Y-%m-%d %H:%M:%S')} 收到数据为{data}")
        response_data, status_code = one_click_service.process(data)
        return jsonify(response_data), status_code

    except Exception as e:
        app.logger.exception("one_click_generate 接口发生未处理异常")
        return jsonify({
            'status': ResponseStatus.ERROR,
            'message': f'内部服务器错误: {e}'
        }), 500


@app.route('/get_user_upload_info', methods=['GET'])
def get_user_upload_info() -> Response:
    """
    获取用户上传统计信息。

    前端页面加载时调用，返回用户的投稿统计数据。
    当前为Mock实现。
    """
    _ = request.args.get('userName')

    return jsonify({
        'status': ResponseStatus.SUCCESS,
        'data': {
            'total_count_today': 0,
            'unprocessed_count_today': 0,
            'remote_upload_count': 0
        }
    })


# =============================================================================
# 应用启动
# =============================================================================

if __name__ == "__main__":
    print("Flask 接口服务启动...")
    app.run(
        host='0.0.0.0',
        port=5002,
        debug=True,
        use_reloader=False
    )
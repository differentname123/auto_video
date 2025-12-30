import asyncio
import hashlib
import time
from pathlib import Path
from typing import Optional, Tuple, List

from utils.common_utils import download_cover_minimal
# --- 保持必要的项目内导入 ---
from third_party.TikTokDownloader.src.tools import beautify_string
from src.config import Parameter, Settings
from src.custom import PROJECT_ROOT
from src.downloader import Downloader
from src.extract import Extractor
from src.interface import Comment, Detail
from src.link import Extractor as LinkExtractor
from src.manager import Database, DownloadRecorder
from src.module import Cookie
from src.record import BaseLogger
from src.tools import ColorfulConsole

# --- 全局常量 ---
COMMENTS_PER_PAGE = 20


# 1. 上下文管理器 (核心依赖)
class DouyinScraperContext:
    """
    一个异步上下文管理器，用于初始化和清理抖音爬取任务所需的资源。
    """

    def __init__(self):
        self.console = ColorfulConsole()
        self.settings = Settings(PROJECT_ROOT, self.console)
        self.config_data = self.settings.read()
        self.database = Database()
        self.params: Optional[Parameter] = None

    async def __aenter__(self) -> Tuple[Parameter, ColorfulConsole]:
        """异步进入上下文，初始化所有资源并返回核心对象。"""
        await self.database.__aenter__()

        # recorder 的 switch 设置为 False，因为我们只是用它来获取评论或单次下载
        recorder = DownloadRecorder(self.database, switch=False, console=self.console)
        cookie_obj = Cookie(self.settings, self.console)

        self.params = Parameter(
            settings=self.settings,
            cookie_object=cookie_obj,
            logger=BaseLogger,
            console=self.console,
            recorder=recorder,
            **self.config_data,
        )

        self.params.set_headers_cookie()
        await self.params.update_params_offline()

        return self.params, self.console

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步退出上下文，确保所有资源被安全释放。"""
        if self.params:
            await self.params.close_client()
        if self.database:
            await self.database.close()
        self.console.info("资源已清理完毕。")




# 3. 核心异步逻辑 (视频/元数据)
async def _async_download_and_get_latest_file(douyin_video_url: str, only_meta=False) -> Optional[Tuple[str, list]]:
    # download_path_str = 'Volume/Download'
    download_path_str = Path(__file__).parent / 'Volume' / 'Download'

    try:
        async with DouyinScraperContext() as (params, console):
            link_extractor = LinkExtractor(params)
            data_extractor = Extractor(params)
            downloader = Downloader(params)

            video_ids = await link_extractor.run(douyin_video_url, type_="detail")
            console.info(f"步骤 1/4: 从 URL 中提取视频 ID...{video_ids}")

            if not video_ids:
                console.error(f"未能从链接 {douyin_video_url} 中提取到有效的视频 ID。")
                return None
            video_id = video_ids[0]

            console.info("\n步骤 2/4: 获取视频元数据...")
            detail_api = Detail(params, detail_id=video_id)
            video_data_raw = await detail_api.run()
            if not video_data_raw:
                console.error(f"获取视频 ID {video_id} 的元数据失败。")
                return None

            class MockRecorder:
                async def save(self, data): pass

                @property
                def field_keys(self): return ["uri", "nickname", "id", "desc"]

            processed_data = await data_extractor.run([video_data_raw], recorder=MockRecorder(), type_="detail")

            # 如果只需要元数据，在此返回
            if only_meta:
                return processed_data  # path 为 None

            if not processed_data:
                console.error("处理视频元数据失败。")
                return None

            # 下载封面逻辑
            for detail_data in processed_data:
                static_cover = detail_data.get('static_cover')
                video_id = detail_data.get('id')
                cover_path = Path(download_path_str) / "cover" / f"{video_id}.jpg"
                await download_cover_minimal(static_cover, cover_path)
                detail_data['abs_cover_path'] = str(cover_path.resolve())

            item = processed_data[0]
            item["desc"] = beautify_string(item["desc"], 64)
            name = downloader.generate_detail_name(item)

            console.info("\n步骤 4/4: 开始下载视频...")
            await downloader.run(processed_data, type_="detail")

            name = f"{name}.mp4"
            latest_file = Path(download_path_str) / name
            print(latest_file.resolve())
            if latest_file.exists():
                console.info("\n下载任务完成！")
                return str(latest_file.resolve()), processed_data
            else:
                console.error(f"下载失败，文件 {latest_file.resolve()} 不存在。")
                return None

    except Exception as e:
        ColorfulConsole().error(f"下载过程中发生未知错误: {e}")
        return None


# 4. 核心异步逻辑 (评论)
async def _async_get_comments(video_id: str, pages_to_fetch: int):
    try:
        async with DouyinScraperContext() as (params, console):
            console.info(f"正在为视频 ID: {video_id} 获取 {pages_to_fetch} 页评论...")
            comment_api = Comment(params, detail_id=video_id, pages=pages_to_fetch)
            comments_data = await comment_api.run()
            return comments_data
    except Exception as e:
        ColorfulConsole().error(f"获取评论过程中发生错误: {e}")
        return None


async def _download_comments_async(text_list, limit=10, video_id=''):
    """异步下载评论图片"""
    result = []
    count = 0

    for text, digg_count, image_urls in text_list:
        count += 1
        final_path = None

        if count <= limit and image_urls:
            name_seed = f"{video_id}_{text}".encode('utf-8')
            file_hash = hashlib.md5(name_seed).hexdigest()
            save_dir = Path('Download') / 'comment_images'
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"{file_hash}.jpg"

            if save_path.exists():
                final_path = save_path.resolve()
            else:
                for url in image_urls:
                    success = await download_cover_minimal(url, save_path)
                    if success:
                        final_path = save_path.resolve()
                        break

        result.append((text, digg_count, str(final_path) if final_path else None))

    return result


def parse_comment(comment: dict) -> tuple:
    """从 comment 字典中安全地提取文本、点赞数和图片URL列表。"""
    if not isinstance(comment, dict):
        return ('无效评论', 0, [])

    text = comment.get('text', '无内容')
    digg_count = comment.get('digg_count', 0)
    image_urls = []
    try:
        image_urls = comment['image_list'][0]['origin_url']['url_list']
        if not isinstance(image_urls, list):
            image_urls = []
    except (KeyError, IndexError, TypeError):
        pass

    return (text, digg_count, image_urls)


# ====================================================================
# 对外接口
# ====================================================================

def download_douyin_video_sync(douyin_video_url: str) -> Optional[Tuple[str, list]]:
    """
    功能 1: 同步下载抖音视频，返回 (文件路径, 元数据列表)
    """
    try:
        return asyncio.run(_async_download_and_get_latest_file(douyin_video_url))
    except RuntimeError as e:
        print(f"错误：不能在一个已经运行的异步事件循环中调用此同步函数。错误信息: {e}")
        return None


def get_meta_info(douyin_video_url: str) -> Optional[Tuple[str, list]]:
    """
    功能 2: 仅获取抖音视频元数据，不下载视频，返回 (None, 元数据列表)
    """
    try:
        return asyncio.run(_async_download_and_get_latest_file(douyin_video_url, only_meta=True))
    except RuntimeError as e:
        print(f"错误：不能在一个已经运行的异步事件循环中调用此同步函数。错误信息: {e}")
        return None


def get_comment(video_id: str, comment_limit=200):
    """
    功能 3: 获取指定视频的评论，并下载评论中的图片（前10张）
    """
    try:
        start_time = time.time()
        print(f"开始获取视频 {video_id} 的前 {comment_limit} 条评论...")

        # 计算需要的页数
        pages_needed = (comment_limit + COMMENTS_PER_PAGE - 1) // COMMENTS_PER_PAGE

        # 获取评论数据
        comments = asyncio.run(_async_get_comments(video_id, pages_to_fetch=pages_needed))

        if not comments:
            return []

        # 按点赞数排序
        comments = sorted(comments, key=lambda x: x.get('digg_count', 0), reverse=True)

        # 解析评论
        text_list = [parse_comment(c) for c in comments]

        # 下载图片 (前10个带图评论)
        text_list = asyncio.run(_download_comments_async(text_list, limit=10, video_id=video_id))

        print(f"{video_id} 获取到 {len(text_list)} 条评论，耗时 {time.time() - start_time:.2f} 秒。")
        return text_list

    except Exception as e:
        print(f"获取评论时发生错误: {e}")
        return []


def main_download():
    """
    演示如何调用封装好的 *同步* 下载函数。
    【演示性修改】: 修改此函数以展示如何处理新的返回值。
    """
    douyin_video_url = "https://www.douyin.com/video/7586268571870612795"
    print(f"开始下载视频: {douyin_video_url}")

    # 接收返回的元组
    result = download_douyin_video_sync(douyin_video_url)

    # 检查结果并解包
    if result:
        downloaded_file_path, metadata = result
        print("\n✅ 下载成功！")
        print(f"   文件已保存至: {downloaded_file_path}")
        # 打印获取到的元数据（通常是一个列表，我们打印第一项）
        if metadata and isinstance(metadata, list):
            print("\n   获取到的元数据:")
            # 使用更安全的方式访问元数据
            first_item = metadata[0] if metadata else {}
            print(f"   - 视频ID (uri): {first_item.get('uri')}")
            print(f"   - 作者昵称 (nickname): {first_item.get('nickname')}")
            print(f"   - 作品ID (id): {first_item.get('id')}")
            print(f"   - 视频描述 (desc): {first_item.get('desc', '无描述')[:50]}...")  # 打印前50个字符
        else:
            print("\n   未获取到有效的元数据。")

    else:
        print("\n❌ 下载失败，请查看上面的日志信息。")


if __name__ == "__main__":
    main_download()
    # get_comment("7535033987593440527", comment_limit=20)
    # gen_title_from_video("test.mp4")
# -- coding: utf-8 --
""":authors:
    zhuxiaohu
:create_date:
    2025/12/12 1:59
:last_date:
    2025/12/12 1:59
:description:
    
"""
import os
import json
import os
import time
import argparse
import sys
from typing import Tuple, Optional
from playwright.sync_api import sync_playwright, Page, expect
import time
import datetime
import sys
import csv
import os
import traceback # 用于捕获更详细的异常信息

from utils.auto_web.web_auto import query_google_ai_studio
from utils.common_utils import read_json, read_file_to_str, save_json, is_valid_target_file_simple


def fun():
    test_prompt = read_file_to_str("视频场景逻辑切分只根据视频内容.txt")

    meta_path = r"W:\project\python_project\watermark_remove\LLM\TikTokDownloader\back_up\metadata_cache.json"
    meta_data = read_json(meta_path)


    # --------- 日志和循环配置 ---------
    # 使用 .jsonl 扩展名以表示 JSON Lines 格式
    LOG_FILE = "stability_test_log.json"
    log_data = read_json(LOG_FILE)
    exist_video_id_list = list(log_data.keys())
    MIN_INTERVAL_SECONDS = 60  # 每次循环的最小间隔时间（秒）
    # ----------------------------------

    print(f"测试已开始，日志将记录到 {LOG_FILE}")
    print(f"每次循环最小间隔为 {MIN_INTERVAL_SECONDS} 秒。")
    print("按 Ctrl+C 停止测试。")

    iteration_count = 0

    for video_id, info in meta_data.items():
        video_path = info.get("video_path", "")
        # if video_id == "7568400808300121354":
        #     print()
        if is_valid_target_file_simple(video_path, 120000) is False:
            print(f"[+] 视频不存在，跳过处理: {video_id}")
            continue
        response_content = log_data[video_id]["response_content"]
        if video_id in exist_video_id_list and "has_overall_bgm" in response_content:
            print(f"[+] 视频已测试过，跳过处理: {video_id}")
            continue

        if "Content blocke" in response_content:
            print(f"[+] 视频被内容屏蔽，跳过处理: {video_id}")
            continue
        test_file = video_path

        iteration_count += 1
        print(f"\n{'=' * 20} 第 {iteration_count} 次测试开始 {'=' * 20}")

        # 1. 记录循环开始时间
        loop_start_time = time.time()

        # 初始化本次循环的结果变量
        error_message = ""
        response_content = ""

        try:
            # 记录函数调用的开始时间
            call_start_time = time.time()

            # 2. 调用核心函数
            err, response = query_google_ai_studio(prompt=test_prompt, file_path=test_file)
            if "You've reached your rate limit. Please try again later" in response:
                print("[-] 检测到速率限制，等待10分钟后重试...")
                time.sleep(600)

            # 记录函数调用的结束时间
            call_end_time = time.time()
            call_duration = call_end_time - call_start_time

            if err:
                status = "FAIL"
                error_message = str(err)
                print(f"❌ 函数返回错误: {error_message}")
            else:
                status = "SUCCESS"
                response_content = response
                print(f"✅ 函数调用成功！")
                # 为了日志整洁，可以只打印部分内容
                print(f"   模型回复 (前50字符): {response[:50]}...")

        except Exception as e:
            # 3. 捕获任何未预料的异常，防止程序崩溃
            call_end_time = time.time()  # 即使出错，也记录时间
            call_duration = call_end_time - call_start_time
            status = "CRASH"
            # 使用 traceback 获取详细的堆栈信息，便于排查问题
            error_message = traceback.format_exc()
            print(f"💥 程序发生严重错误 (CRASH): {e}")
            print("   详细堆栈信息已记录到日志。")

        # 4. 准备写入日志的数据字典
        current_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {
            "timestamp": current_timestamp,
            "duration_seconds": float(f"{call_duration:.2f}"),
            "status": status,
            "error_message": error_message,
            "response_content": response_content
        }
        log_data[f"{video_id}"] = log_entry

        # 5. 将结果字典转换为JSON字符串并追加写入文件
        try:
            # 使用 'a' 模式以追加方式打开文件
            # ensure_ascii=False 保证中文字符能被正确写入，而不是被转义
            save_json(LOG_FILE, log_data)

            print(f"结果已保存到 {LOG_FILE}")
        except IOError as e:
            print(f"!!!!!! 严重: 无法写入日志文件 {LOG_FILE}: {e} !!!!!!")

        # 6. 控制循环间隔，确保至少为1分钟
        loop_end_time = time.time()
        elapsed_time = loop_end_time - loop_start_time

        if elapsed_time < MIN_INTERVAL_SECONDS:
            wait_time = MIN_INTERVAL_SECONDS - elapsed_time
            print(f"本次循环耗时 {elapsed_time:.2f} 秒，等待 {wait_time:.2f} 秒后开始下一次测试...")
            time.sleep(wait_time)
        else:
            print(f"本次循环耗时 {elapsed_time:.2f} 秒，已超过最小间隔，立即开始下一次测试。")



if __name__ == '__main__':
    fun()
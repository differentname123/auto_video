from flask import Flask, request, jsonify, render_template

app = Flask(__name__)


# ==============================================================================
# 1. 核心业务接口
# ==============================================================================

@app.route('/one-click-generate', methods=['POST'])
def one_click_generate():
    """
    接收前端提交的任务请求
    只负责：接收数据 -> 解析合并参数 -> 立即返回
    """
    try:
        # 1. 获取前端 JSON 数据
        data = request.get_json()
        if not data:
            return jsonify({'status': 'error', 'message': '请求体为空'}), 400

        # 2. 提取顶层参数
        user_name = data.get('userName')
        global_settings = data.get('global_settings', {})  # 全局通用设置
        video_list = data.get('video_list', [])  # 视频列表

        # 简单的权限/非空校验
        if not user_name:
            return jsonify({'status': 'error', 'message': '未提供用户名'}), 400
        if not video_list:
            return jsonify({'status': 'error', 'message': '视频列表为空'}), 400

        print(f"收到请求 | 用户: {user_name} | 视频数: {len(video_list)}")

        # 3. 遍历视频列表，进行配置适配 (Adapter Logic)
        # 这里虽然没有实际执行，但展示了如何将前端的 Global + Local 配置合并
        processed_tasks = []

        for video_item in video_list:
            url = video_item.get('url')
            if not url: continue

            # --- 关键：合并配置 ---
            # 以全局设置(global_settings)为基础，覆盖单个视频的特定设置
            final_task_config = global_settings.copy()

            # 将前端 video_list 里的独立开关合并进去
            final_task_config.update({
                'video_url': url,
                'has_author_voice': video_item.get('has_author_voice'),  # 前端: select-voice
                'is_realtime': video_item.get('is_realtime'),  # 前端: select-realtime
                'need_text': video_item.get('need_text'),  # 前端: select-text
                'need_emoji': video_item.get('need_emoji')  # 前端: select-emoji
            })

            # 这里可以调用你的业务函数，或者存入数据库
            # 例如: database.save(final_task_config)
            # 或者: thread_pool.submit(process_video, final_task_config)
            processed_tasks.append(final_task_config)

        # 4. 打印一下合并后的第一个任务配置，确认适配成功
        if processed_tasks:
            print(f"任务配置示例 (First Item): {processed_tasks[0]}")

        # 5. 立即返回成功，实现“快速接收”
        return jsonify({
            'status': 'success',
            'message': f'服务端已接收 {len(processed_tasks)} 个任务'
        })

    except Exception as e:
        print(f"接口异常: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


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
    app.run(host='0.0.0.0', port=5002, debug=True, threaded=True)
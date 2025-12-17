# -- coding: utf-8 --
""":authors:
    zhuxiaohu
:create_date:
    2025/12/15 18:13
:last_date:
    2025/12/15 18:13
:description:
    
"""
import os

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
import torch
import torchaudio
from pprint import pprint

# 1. 加载模型和工具
# 第一次运行时会自动从网上下载模型
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True)  # 设置为True确保下载最新版本

# 从utils中获取我们需要的函数
(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils

# 2. 准备示例音频
# Silero VAD 期望的音频格式是：16kHz, 单声道, 16-bit PCM
# 你可以使用自己的WAV文件，或下载一个示例
SAMPLING_RATE = 16000  # Silero VAD 默认采样率
wav_file = 'temp/test.mp3'

# 使用 torchaudio 读取音频
# 注意：它返回的是一个tensor，这正是模型需要的格式
audio_tensor = read_audio(wav_file, sampling_rate=SAMPLING_RATE)

# 3. 进行语音检测
# `get_speech_timestamps` 是最常用的函数，我们将在下一节详细讲解
# 这里我们先手动遍历音频块来理解基本原理
speech_probs = []
window_size_samples = 512  # VAD模型处理的窗口大小（以采样点计）

# 遍历整个音频张量
for i in range(0, len(audio_tensor), window_size_samples):
    chunk = audio_tensor[i: i + window_size_samples]
    if len(chunk) < window_size_samples:
        break  # 如果最后一块不够长，则忽略

    # `model()` 是核心的调用方法
    speech_prob = model(chunk, SAMPLING_RATE).item()
    speech_probs.append(speech_prob)

# `model.reset_states()` 用于重置模型的内部状态，处理新文件前调用
model.reset_states()

print("前10个音频块的语音置信度：")
pprint(speech_probs[:10])

# 你可以看到，输出是一系列0到1之间的浮点数，数值越高代表是语音的可能性越大
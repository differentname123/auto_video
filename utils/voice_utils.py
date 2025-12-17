# 修正版：使用 silero_vad 的当前接口，并包含能量微调与容错
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
import librosa
import numpy as np
import sys

AUDIO_PATH = 'temp/vocals.wav'   # <-- 确保和下面 librosa.load 使用的是同一文件

# 1) load model (不要解包)
model = load_silero_vad()

# 2) read audio for VAD using package helper (silero expects 16k mono)
wav = read_audio(AUDIO_PATH)   # 返回 numpy array (float32)，采样率已变为 16k（silero 内部做了处理）
# 若 read_audio 返回 torch.Tensor，则转为 numpy（下面做了兼容）
if hasattr(wav, "numpy"):
    try:
        wav = wav.numpy()
    except Exception:
        wav = np.asarray(wav)

# 3) get speech timestamps (秒)
speech_ts = get_speech_timestamps(wav, model, return_seconds=True)

if not speech_ts:
    print("No speech segments detected.")
    sys.exit(0)

# 4) 合并短停顿为“句”（示例阈值 0.35s）
records = []
pause_thresh = 0.35
cur_start = speech_ts[0]['start']
cur_end = speech_ts[0]['end']
for s in speech_ts[1:]:
    if s['start'] - cur_end >= pause_thresh:
        records.append({'start': cur_start, 'end': cur_end})
        cur_start = s['start']
        cur_end = s['end']
    else:
        cur_end = s['end']
records.append({'start': cur_start, 'end': cur_end})

# 5) 用 librosa 重新加载音频用于短时能量微调（保证 sr=16000）
y, sr = librosa.load(AUDIO_PATH, sr=16000, mono=True)
hop = 0.01                        # 帧移 10ms
frame_len = int(0.02 * sr)        # 帧长 20ms

for r in records:
    center = int(r['end'] * sr)
    w = int(0.2 * sr)             # ±0.2s 搜索窗口
    start = max(0, center - w)
    stop = min(len(y), center + w)
    seg = y[start:stop]

    # 如果片段太短，直接用原 end
    if len(seg) < frame_len + 1:
        r['end_refined'] = r['end']
        continue

    # 将段分帧并计算 short-time energy
    hop_n = int(hop * sr)
    try:
        frames = librosa.util.frame(seg, frame_length=frame_len, hop_length=hop_n).T
    except Exception:
        # 防御：若 frame 失败（长度等问题），直接不调整
        r['end_refined'] = r['end']
        continue

    ste = (frames ** 2).sum(axis=1)
    min_i = int(np.argmin(ste))
    adj_sample = start + min_i * hop_n
    adj_time = adj_sample / sr
    # 保证微调后不超过原来 end + 0.3s 或小于 start
    adj_time = max(r['start'], min(adj_time, r['end'] + 0.3))
    r['end_refined'] = adj_time

# 输出结果
for i, r in enumerate(records):
    print(f"segment {i}: start={r['start']:.3f}s  end_raw={r['end']:.3f}s  end_refined={r.get('end_refined', r['end']):.3f}s")

import subprocess
import librosa
import numpy as np
import os


def analyze_audio_purity(video_path):
    temp_audio = "temp_audio_sample.wav"
    max_duration=120
    try:
        # 1. 使用 FFmpeg 提取音频
        # -t: 限制持续时间 (这里设置为 60秒)
        # -vn: 禁用视频
        # -ar: 采样率固定 44100
        # -ac: 单声道 (混合声道更容易发现整体噪点)
        command = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-vn',
            '-t', str(max_duration),  # <--- 核心修改：只截取前60秒
            '-ar', '44100',
            '-ac', '1',
            temp_audio
        ]

        # 执行命令，捕获错误以便调试
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

        # 2. 使用 librosa 加载音频
        # 即使 ffmpeg 没切准，这里也可以加 duration 参数作为双重保险
        if not os.path.exists(temp_audio):
            return {"error": "音频提取失败，临时文件未生成"}

        y, sr = librosa.load(temp_audio, sr=44100)

        # 防止空音频或极短音频导致的计算错误
        if len(y) < sr * 0.5:  # 只有不到0.5秒
            return {"error": "音频过短，无法有效分析"}

        # 3. 计算频谱平坦度 (Spectral Flatness)
        # 越接近 1，表示越像白噪声（沙沙声）
        flatness = librosa.feature.spectral_flatness(y=y)
        avg_flatness = np.mean(flatness)

        # 4. 计算高频能量占比 (Roll-off Frequency)
        # 刺耳声音通常会导致截止频率升高
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
        avg_rolloff = np.mean(rolloff)

        # 5. 计算静音部分的底噪 (核心检测点)
        # 计算均方根能量 (RMS)
        rms = librosa.feature.rms(y=y)
        # 找到能量最低的 20% 片段（认为是静音或说话间隙）
        # 如果前1分钟全是说话，这个指标可能会略高，但在同类视频对比中依然公平
        low_energy_threshold = np.percentile(rms, 20)
        low_energy_idx = np.where(rms <= low_energy_threshold)

        if len(low_energy_idx[1]) > 0:
            # 只取低能量片段的平坦度平均值
            noise_floor_flatness = np.mean(flatness[:, low_energy_idx[1]])
        else:
            noise_floor_flatness = avg_flatness

        # 6. 综合评分 (逻辑：平坦度权重 + 高频权重)
        # 这个系数是经验值，用来放大差异
        purity_score = (avg_flatness * 1000) + (avg_rolloff / 500) + (noise_floor_flatness * 500)

        return {
            "file": os.path.basename(video_path),
            "avg_flatness": float(avg_flatness),
            "avg_rolloff_hz": float(avg_rolloff),
            "noise_floor_flatness": float(noise_floor_flatness),
            "impurity_index": float(purity_score)
        }

    except subprocess.CalledProcessError as e:
        # 捕获 FFmpeg 报错
        return {"error": f"FFmpeg error: {e.stderr.decode('utf-8', errors='ignore')}"}
    except Exception as e:
        return {"error": str(e)}

    finally:
        if os.path.exists(temp_audio):
            os.remove(temp_audio)


if __name__ == "__main__":
    video_file_list = [r"W:\project\python_project\auto_video\videos\material\7492752531630705939\7492752531630705939_origin.mp4",
                       r"W:\project\python_project\auto_video\videos\material\7492752531630705939\7492752531630705939_static_cut.mp4",
                       r"W:\project\python_project\auto_video\videos\material\7492752531630705939\7492752531630705939_image_text.mp4",
                       r"W:\project\python_project\auto_video\videos\task\7492752531630705939\6968c05fe22dbe4a9bd482ed\all_scene.mp4",
                       r"W:\project\python_project\auto_video\videos\task\7492752531630705939\6968c05fe22dbe4a9bd482ed\title.mp4",
                       r"W:\project\python_project\auto_video\videos\task\7492752531630705939\6968c05fe22dbe4a9bd482ed\ending.mp4",
                       r"W:\project\python_project\auto_video\videos\task\7492752531630705939\6968c05fe22dbe4a9bd482ed\watermark.mp4",
                       r"W:\project\python_project\auto_video\videos\task\7492752531630705939\6968c05fe22dbe4a9bd482ed\final_remake.mp4",
                       r"W:\project\python_project\auto_video\videos\task\7492752531630705939\6968c05fe22dbe4a9bd482ed\final_remake - 副本.mp4"

                       ]
    for video_file in video_file_list:
        result = analyze_audio_purity(video_file)
        video_name = os.path.basename(video_file)

        if "error" not in result:
            print(f"\n\n分析结果:{video_name}")
            print(f"  频谱平坦度 (0-1): {result['avg_flatness']:.6f} (越大沙沙声越重)")
            print(f"  高频截止点 (Hz): {result['avg_rolloff_hz']:.2f}")
            print(f"  底噪平坦度: {result['noise_floor_flatness']:.6f}")
            print(f"  综合杂质指数: {result['impurity_index']:.2f}")
        else:
            print(f"错误: {result['error']}")

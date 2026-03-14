#!/usr/bin/env python3
"""
ClipForge Backend — Gaming Stream Auto-Clipper
Optimized for Railway deployment (no torch/whisper)
Detects: gunshots, explosions, voice spikes, killstreaks
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import yt_dlp
import librosa
import numpy as np
import tempfile
import os
import json
import threading
import uuid
import subprocess
import shutil

app = Flask(__name__)
CORS(app)
@app.route('/')
def home():
    return jsonify({'status': 'ClipForge running'})

JOBS = {}
CLIPS_DIR = tempfile.mkdtemp()

try:
    import cv2
    CV2 = True
except:
    CV2 = False

# ── ROUTES ────────────────────────────────────────────────────────────────────

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'opencv': CV2, 'ffmpeg': shutil.which('ffmpeg') is not None})

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    url = data.get('url', '').strip()
    if not url:
        return jsonify({'error': 'No URL'}), 400
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {'status': 'queued', 'progress': 0, 'step': 'Starting...', 'highlights': []}
    threading.Thread(target=pipeline, args=(job_id, url, data), daemon=True).start()
    return jsonify({'job_id': job_id})

@app.route('/status/<job_id>')
def status(job_id):
    return jsonify(JOBS.get(job_id, {'error': 'not found'}))

@app.route('/clip', methods=['POST'])
def make_clip():
    """Cut a clip from downloaded video with optional text overlay"""
    data = request.json
    job_id = data.get('job_id')
    start = data.get('start', 0)
    end = data.get('end', 30)
    fmt = data.get('format', 'video')       # 'video' or 'shorts'
    text = data.get('text', '')             # overlay text
    text_style = data.get('text_style', 'modern')  # modern, fire, minimal

    job = JOBS.get(job_id, {})
    video_path = job.get('video_path', '')

    if not video_path or not os.path.exists(video_path):
        return jsonify({'error': 'Video not found — re-analyze first'}), 400

    clip_id = str(uuid.uuid4())[:8]
    out_path = os.path.join(CLIPS_DIR, f'clip_{clip_id}.mp4')

    # Build ffmpeg command
    duration = end - start

    # Scale filter based on format
    if fmt == 'shorts':
        # 9:16 vertical — crop center
        scale = 'crop=ih*9/16:ih,scale=1080:1920'
    else:
        # 16:9 horizontal
        scale = 'scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2'

    # Text overlay filters
    text_filter = ''
    if text.strip():
        safe_text = text.replace("'", "").replace('"', '').replace(':', '\\:')[:60]
        if text_style == 'modern':
            text_filter = (
                f",drawtext=text='{safe_text}'"
                f":fontsize=64:fontcolor=white:x=(w-text_w)/2:y=h-150"
                f":borderw=3:bordercolor=black@0.8"
                f":box=1:boxcolor=black@0.5:boxborderw=12"
            )
        elif text_style == 'fire':
            text_filter = (
                f",drawtext=text='{safe_text}'"
                f":fontsize=72:fontcolor=orange:x=(w-text_w)/2:y=h-160"
                f":borderw=4:bordercolor=red@0.9"
            )
        elif text_style == 'minimal':
            text_filter = (
                f",drawtext=text='{safe_text}'"
                f":fontsize=48:fontcolor=white@0.9:x=(w-text_w)/2:y=h-120"
                f":borderw=2:bordercolor=black@0.6"
            )
        elif text_style == 'top':
            text_filter = (
                f",drawtext=text='{safe_text}'"
                f":fontsize=64:fontcolor=white:x=(w-text_w)/2:y=60"
                f":borderw=3:bordercolor=black@0.8"
                f":box=1:boxcolor=black@0.6:boxborderw=10"
            )

    vf = f"{scale}{text_filter}"

    cmd = [
        'ffmpeg', '-y',
        '-ss', str(start),
        '-i', video_path,
        '-t', str(duration),
        '-vf', vf,
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '23',
        '-c:a', 'aac',
        '-b:a', '128k',
        '-movflags', '+faststart',
        out_path
    ]

    try:
        subprocess.run(cmd, capture_output=True, timeout=120)
        if os.path.exists(out_path):
            return send_file(out_path, mimetype='video/mp4',
                           as_attachment=True,
                           download_name=f'clip_{fmt}_{clip_id}.mp4')
        else:
            return jsonify({'error': 'FFmpeg failed to create clip'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── PIPELINE ──────────────────────────────────────────────────────────────────

def pipeline(job_id, url, opts):
    try:
        tmpdir = tempfile.mkdtemp()

        upd(job_id, 5, '📥 Downloading stream...')
        audio_path, video_path, duration = download(url, tmpdir)
        JOBS[job_id]['video_path'] = video_path
        JOBS[job_id]['duration'] = duration
        upd(job_id, 22, f'✓ Downloaded ({int(duration)}s)')

        upd(job_id, 28, '🎮 Detecting gaming events (gunshots, explosions, voice)...')
        moments = detect_gaming_moments(audio_path, duration)
        upd(job_id, 65, f'✓ Found {len(moments)} candidate moments')

        if CV2 and video_path and os.path.exists(video_path):
            upd(job_id, 68, '🎥 Analyzing video frames...')
            moments = boost_with_video(video_path, moments)
            upd(job_id, 82, '✓ Video analysis done')

        upd(job_id, 88, '📊 Ranking highlights...')
        highlights = rank(moments, duration)

        JOBS[job_id].update({
            'status': 'done', 'progress': 100,
            'step': f'✓ {len(highlights)} highlights found',
            'highlights': highlights
        })

    except Exception as e:
        JOBS[job_id].update({'status': 'error', 'step': str(e)})


def upd(job_id, pct, step):
    JOBS[job_id].update({'progress': pct, 'step': step, 'status': 'running'})


# ── DOWNLOAD ──────────────────────────────────────────────────────────────────

def download(url, tmpdir):
    audio_path, video_path = None, None

    # Audio
    ydl_a = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(tmpdir, 'audio.%(ext)s'),
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav'}],
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_a) as ydl:
        info = ydl.extract_info(url, download=True)
        duration = info.get('duration', 0) or 0

    for f in os.listdir(tmpdir):
        if f.endswith('.wav'):
            audio_path = os.path.join(tmpdir, f)

    # Video (low res for analysis + clipping source)
    vpath = os.path.join(tmpdir, 'video.mp4')
    ydl_v = {
        'format': 'bestvideo[height<=720][ext=mp4]+bestaudio/best[height<=720]',
        'outtmpl': vpath,
        'merge_output_format': 'mp4',
        'quiet': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_v) as ydl:
            ydl.download([url])
        if os.path.exists(vpath):
            video_path = vpath
    except:
        pass

    return audio_path, video_path, duration


# ── GAMING AUDIO DETECTION ────────────────────────────────────────────────────

def detect_gaming_moments(audio_path, total_dur):
    """
    Tuned for Arc Raiders / FPS games:
    - Gunshots: sharp transient spikes (high onset, short duration)
    - Explosions: sustained low-frequency burst
    - Voice reactions: mid-frequency sudden energy
    - Killstreaks: multiple rapid transients close together
    """
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    dur = librosa.get_duration(y=y, sr=sr)
    hop = 512

    # 1. Onset strength — detects all sudden sounds (gunshots, explosions)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    onset_t = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr, hop_length=hop)

    # 2. RMS energy — overall loudness
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    rms_t = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)

    # 3. Spectral centroid — voice vs gunshot distinction
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
    centroid_t = librosa.frames_to_time(np.arange(len(centroid)), sr=sr, hop_length=hop)

    # 4. Low frequency energy — explosions and heavy weapons
    # Filter to 20-300Hz range
    y_low = librosa.effects.preemphasis(y, coef=-0.97)  # boost lows
    rms_low = librosa.feature.rms(y=y_low, frame_length=4096, hop_length=hop)[0]

    # 5. Onset density — rapid fire / killstreak detection
    # Count onsets in rolling 2s window
    onset_peaks = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop, units='time')

    def norm(x):
        mn, mx = x.min(), x.max()
        return (x - mn) / (mx - mn + 1e-9)

    onset_n = norm(onset_env)
    rms_n = norm(rms)
    centroid_n = norm(centroid)
    rms_low_n = norm(rms_low)

    # Scan 3-second windows
    window = 3
    moments = []
    num_w = max(1, int(dur / window))

    for i in range(num_w):
        t0, t1 = i * window, (i + 1) * window

        def avg(times, vals):
            mask = (times >= t0) & (times < t1)
            return float(vals[mask].mean()) if mask.any() else 0.0

        o = avg(onset_t, onset_n)
        r = avg(rms_t, rms_n)
        c = avg(centroid_t, centroid_n)
        l = avg(rms_t, rms_low_n)

        # Count rapid fire onsets in this window (killstreak detection)
        rapid = len([p for p in onset_peaks if t0 <= p < t1])
        rapid_score = min(1.0, rapid / 8.0)  # 8+ shots = max score

        # Classify event type
        event_type = 'hype'
        reasons = []

        # Gunshot pattern: very high onset, short burst
        if o > 0.75 and r > 0.5:
            event_type = 'gunshot'
            reasons.append('gunshot detected')

        # Explosion: high onset + high low freq
        if o > 0.6 and l > 0.65:
            event_type = 'explosion'
            reasons.append('explosion/heavy weapon')

        # Voice reaction: high centroid + sudden rms spike
        if c > 0.7 and r > 0.55:
            event_type = 'voice'
            reasons.append('voice reaction')

        # Killstreak: rapid multiple onsets
        if rapid >= 5:
            event_type = 'killstreak'
            reasons.append(f'rapid fire ({rapid} shots)')

        # Combined score — weighted for gaming
        score = (
            o * 0.35 +
            r * 0.25 +
            l * 0.15 +
            rapid_score * 0.25
        ) * 100

        if score > 45:
            moments.append({
                'time_sec': t0,
                'end_sec': min(t0 + 35, dur),
                'audio_score': round(score, 1),
                'event_type': event_type,
                'reason': ', '.join(reasons) if reasons else 'audio event',
                'rapid_count': rapid,
                'signals': {'onset': round(o,3), 'rms': round(r,3), 'lowfreq': round(l,3), 'rapid': rapid}
            })

    # Deduplicate — keep best, min 20s apart
    moments.sort(key=lambda x: x['audio_score'], reverse=True)
    filtered = []
    for m in moments:
        if all(abs(m['time_sec'] - f['time_sec']) > 20 for f in filtered):
            filtered.append(m)
        if len(filtered) >= 12:
            break

    return filtered


# ── VIDEO BOOST ───────────────────────────────────────────────────────────────

def boost_with_video(video_path, moments):
    if not CV2:
        return moments
    cap = cv2.VideoCapture(video_path)
    enhanced = []
    for m in moments:
        t = m['time_sec']
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret1, f1 = cap.read()
        cap.set(cv2.CAP_PROP_POS_MSEC, (t + 0.5) * 1000)
        ret2, f2 = cap.read()

        bonus = 0
        if ret1 and f1 is not None:
            g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
            brightness = float(np.mean(g1))
            if brightness > 160: bonus += 12  # flash/explosion

            if ret2 and f2 is not None:
                g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
                diff = float(np.mean(cv2.absdiff(g1, g2)))
                if diff > 20: bonus += 10  # high motion

        enhanced.append({**m, 'audio_score': min(100, m['audio_score'] + bonus)})
    cap.release()
    return enhanced


# ── RANK ──────────────────────────────────────────────────────────────────────

def rank(moments, duration):
    if not moments:
        return []

    scores = [m['audio_score'] for m in moments]
    mn, mx = min(scores), max(scores)
    spread = mx - mn if mx != mn else 1

    titles = {
        'gunshot':   ['Shots fired!', 'Clean shots', 'Gunfight detected', 'Combat moment'],
        'explosion': ['BOOM! Explosion', 'Heavy weapon fired', 'Big blast detected'],
        'voice':     ['Streamer reacts!', 'Hype reaction', 'Voice spike detected'],
        'killstreak':['KILLSTREAK!', 'Rapid fire sequence', 'Multi-kill moment'],
        'hype':      ['Highlight moment', 'Action detected', 'Big moment'],
    }

    import random
    result = []
    for m in moments:
        final = int(65 + ((m['audio_score'] - mn) / spread) * 33)
        final = min(98, final)
        t, e = int(m['time_sec']), int(m.get('end_sec', m['time_sec'] + 30))
        et = m.get('event_type', 'hype')
        result.append({
            'final_score': final,
            'title': random.choice(titles.get(et, titles['hype'])),
            'event_type': et,
            'tag': et,
            'start_sec': t,
            'end_sec': e,
            'duration': e - t,
            'start_fmt': fmt(t),
            'end_fmt': fmt(e),
            'clip_reason': m.get('reason', 'audio event'),
            'signals': m.get('signals', {}),
            'rapid_count': m.get('rapid_count', 0),
        })

    result.sort(key=lambda x: x['final_score'], reverse=True)
    return result[:8]


def fmt(s):
    s = int(s)
    return f"{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}"


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n🎮 ClipForge Backend — Gaming Stream Clipper")
    print(f"   OpenCV: {'✓' if CV2 else '✗'}")
    print(f"   FFmpeg: {'✓' if shutil.which('ffmpeg') else '✗'}")
    print(f"   Running on port {port}\n")
    app.run(host='0.0.0.0', port=port, debug=False)

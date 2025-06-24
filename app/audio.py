import os
import shutil
import librosa
import numpy as np
import tensorflow as tf
import pickle
import soundfile as sf
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, Depends, Query, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import speech_recognition as sr
from tempfile import mkdtemp
from pydub import AudioSegment
import logging
import asyncio
import joblib
import json
import base64
from io import BytesIO
import wave
import webrtcvad  # type: ignore
import uuid
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# WebRTC Voice Activity Detection (0=less aggressive, 3=most aggressive)
vad = webrtcvad.Vad(2)

# Path setup for ffmpeg and ffprobe (relative, for Deta Space/serverless compatibility)
ffmpeg_path = "./bin/ffmpeg"
ffprobe_path = "./bin/ffprobe"
AudioSegment.converter = ffmpeg_path
AudioSegment.ffmpeg = ffmpeg_path
AudioSegment.ffprobe = ffprobe_path  # type: ignore

# Emotions List
EMOTIONS = ['fear', 'calm', 'neutral', 'angry', 'happy', 'surprise', 'disgust', 'sad']
NEG_EMOS = ['fear', 'angry', 'disgust', 'sad']

# FastAPI app initialization
app = FastAPI()

# Mount static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

# Pastikan folder untuk upload audio tersedia
UPLOAD_DIR = "uploaded_audio"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Global variables for ML model and encoders
model = None
scaler = None
encoder = None
label_list = None

# Load model, scaler, encoder at startup (hanya sekali)
try:
    MODEL_PATH = os.path.join("model", "best_model.h5")
    SCALER_PATH = os.path.join("model", "scaler4.pkl")
    ENCODER_PATH = os.path.join("model", "label_encoder4.pkl")
    
    model = tf.keras.models.load_model(MODEL_PATH) # type: ignore
    scaler = joblib.load(SCALER_PATH)
    with open(ENCODER_PATH, 'rb') as f:
        encoder = pickle.load(f)
    label_list = encoder.categories_[0]
    logger.info("Model and preprocessors loaded successfully")
except Exception as e:
    logger.error(f"Error loading model or preprocessors: {e}")
    raise


@app.get("/")
async def index():
    return FileResponse("static/index.html")

@app.get("/analyze")
async def analyze():
    return FileResponse("static/analyze.html")

@app.get("/aboutus")
async def aboutus():
    return FileResponse("static/aboutus.html")

def extract_features(data, sample_rate):
    """Extract audio features for emotion recognition"""
    try:
        result = np.array([])
        
        # Zero Crossing Rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
        result = np.hstack((result, zcr))
        
        # Chroma
        stft = np.abs(librosa.stft(data))
        chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        result = np.hstack((result, chroma_stft))
        
        # MFCC
        mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mfcc))
        
        # RMS
        rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
        result = np.hstack((result, rms))
        
        # Mel Spectrogram
        mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mel))
        
        return result
    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        return None

def prepare_for_model(X):
    """Prepare features for model prediction"""
    try:
        X_scaled = scaler.transform([X])[0] # type: ignore
        return np.expand_dims(X_scaled, axis=(0,2))
    except Exception as e:
        logger.error(f"Error preparing features for model: {e}")
        return None

def transcribe_audio_data(audio_data, sample_rate):
    """Transcribe audio data to text"""
    try:
        # Create temporary WAV file
        temp_dir = mkdtemp()
        temp_wav = os.path.join(temp_dir, "temp_audio.wav")
        
        # Save audio data as WAV
        sf.write(temp_wav, audio_data, sample_rate)
        
        # Transcribe
        recognizer = sr.Recognizer()
        with sr.AudioFile(temp_wav) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.1) # type: ignore
            audio = recognizer.record(source)
        
        try:
            text = recognizer.recognize_google(audio, language='id-ID') # type: ignore
        except:
            try:
                text = recognizer.recognize_google(audio, language='en-US') # type: ignore
            except:
                text = "[Unrecognized speech]"
                
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
        return text
        
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        return "[Transcription error]"

def process_audio_chunk(audio_data, sample_rate=16000):
    """Process audio chunk for emotion recognition"""
    try:
        # Ensure minimum length
        if len(audio_data) < sample_rate * 0.5:  # At least 0.5 seconds
            return None
            
        # Extract features
        features = extract_features(audio_data, sample_rate)
        if features is None or features.shape[0] != 162:
            return None
        
        # Predict emotion
        x = prepare_for_model(features)
        if x is None:
            return None
            
        pred = model.predict(x, verbose=0) # type: ignore
        emo_idx = np.argmax(pred)
        emo_label = label_list[emo_idx] # type: ignore
        emo_conf = float(np.max(pred))
        
        # Transcribe
        transcript = transcribe_audio_data(audio_data, sample_rate)
        
        return {
            "emotion": emo_label,
            "confidence": round(emo_conf, 3),
            "transcript": transcript,
            "duration": len(audio_data) / sample_rate
        }
        
    except Exception as e:
        logger.error(f"Error processing audio chunk: {e}")
        return None
    
def frame_generator(audio_data, sample_rate=16000, frame_duration_ms=30):
    """Yield successive frames from the audio data array."""
    frame_size = int(sample_rate * frame_duration_ms / 1000)
    offset = 0
    while offset + frame_size <= len(audio_data):
        yield audio_data[offset:offset + frame_size]
        offset += frame_size

def is_silence(audio_data, sample_rate=16000, vad=vad, silence_frames_threshold=15):
    """Return True if enough consecutive frames are non-speech."""
    num_silence = 0
    frame_size = int(sample_rate * 30 / 1000)
    for frame in frame_generator(audio_data, sample_rate, 30):
        # Convert float32 (-1.0..1.0) to int16 PCM
        pcm_bytes = (np.clip(frame, -1, 1) * 32768).astype(np.int16).tobytes()
        if not vad.is_speech(pcm_bytes, sample_rate):
            num_silence += 1
        else:
            num_silence = 0  # reset if speech detected
        if num_silence >= silence_frames_threshold:
            return True
    return False




# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.audio_buffers: dict[WebSocket, list] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.audio_buffers[websocket] = []
        logger.info("WebSocket connected")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if websocket in self.audio_buffers:
            del self.audio_buffers[websocket]
        logger.info("WebSocket disconnected")

    async def send_result(self, websocket: WebSocket, data: dict):
        try:
            await websocket.send_text(json.dumps(data))
        except Exception as e:
            logger.error(f"Error sending WebSocket message: {e}")

manager = ConnectionManager()

import logging

logger = logging.getLogger(__name__)

@app.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    sample_rate = 16000
    vad = webrtcvad.Vad(2)
    buffer = []
    min_samples = sample_rate * 1  # minimal 1 detik buffer
    window_timeout = 2.0  # detik, ganti sesuai kebutuhan
    last_emit_time = datetime.now()
    emitted_any = False

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            now = datetime.now()

            if message["type"] == "audio_chunk":
                # decode audio
                audio_bytes = base64.b64decode(message["data"])
                audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                buffer.extend(audio_data)

                # Jangan proses jika buffer terlalu kecil
                if len(buffer) < min_samples:
                    continue

                # Ambil tail buffer (untuk VAD & timeout window)
                tail_samples = int(sample_rate * 0.9)
                last_part = np.array(buffer[-tail_samples:]) if len(buffer) > tail_samples else np.array(buffer)

                # === 1. Segmentasi Berdasarkan Silence (VAD) ===
                silence = is_silence(last_part, sample_rate, vad=vad, silence_frames_threshold=10)
                timeout = (now - last_emit_time).total_seconds() > window_timeout

                if silence or timeout:
                    # Potong buffer untuk segmen yang akan dianalisis
                    segment_data = np.array(buffer)
                    if len(segment_data) > int(sample_rate * 0.5):
                        result = process_audio_chunk(segment_data, sample_rate)
                        if result:
                            await manager.send_result(websocket, {
                                "type": "analysis_result",
                                "data": result
                            })
                            emitted_any = True
                    buffer = []
                    last_emit_time = datetime.now()

            elif message["type"] == "stop_recording":
                if len(buffer) > int(sample_rate * 0.5):
                    audio_np = np.array(buffer)
                    result = process_audio_chunk(audio_np, sample_rate)
                    if result:
                        await manager.send_result(websocket, {
                            "type": "analysis_result",
                            "data": result
                        })
                buffer = []
                break  # end connection after stop

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


@app.post("/upload-audio")
async def upload_audio(file: UploadFile = File(...)):
    # Pastikan filename pasti string, meski None
    filename_input = file.filename or ""
    ext = os.path.splitext(filename_input)[1].lower()
    if ext not in [".webm", ".wav", ".mp3"]:
        return JSONResponse({"error": "Format tidak didukung"}, status_code=400)

    filename = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    file_url = f"/uploads/{filename}"
    return {"url": file_url}


@app.post("/analyze-audio/")
async def analyze_audio(file: UploadFile = File(...)):
    """Analyze uploaded audio file"""
    temp_dir = None
    try:
        temp_dir = mkdtemp()
        file_path = os.path.join(temp_dir, file.filename or "uploaded_audio")
        
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        size = os.path.getsize(file_path)
        logger.info(f"Received file: {file.filename}, size={size} bytes")

        # Skip very small files
        if size < 5000:
            return JSONResponse({
                "n_segments": 0, 
                "emotion_stats": {}, 
                "segments": [],
                "message": "File too small to process"
            })

        # Load audio directly with librosa (supports many formats)
        try:
            y, sr = librosa.load(file_path, sr=16000)
            
            if len(y) < 8000:  # Less than 0.5 seconds
                return JSONResponse({
                    "n_segments": 0, 
                    "emotion_stats": {}, 
                    "segments": [],
                    "message": "Audio too short"
                })
                
        except Exception as e:
            logger.error(f"Error loading audio: {e}")
            return JSONResponse({
                "error": f"Failed to load audio file: {str(e)}"
            }, status_code=400)

        # Segment audio by silence
        try:
            intervals = librosa.effects.split(y, top_db=25)
            segments = []
            
            for idx, (start, end) in enumerate(intervals):
                seg_y = y[start:end]
                duration = len(seg_y) / sr
                
                if duration >= 0.5:  # At least 0.5 seconds
                    segments.append((seg_y, start/sr, end/sr))
            
            # If no segments found, use whole audio
            if not segments and len(y) > 8000:
                segments.append((y, 0, len(y)/sr))
                
        except Exception as e:
            logger.error(f"Error segmenting audio: {e}")
            return JSONResponse({
                "error": f"Failed to segment audio: {str(e)}"
            }, status_code=400)

        if not segments:
            return JSONResponse({
                "n_segments": 0, 
                "emotion_stats": {}, 
                "segments": [],
                "message": "No valid speech segments found"
            })

        # Process each segment
        results = []
        for i, (seg_y, t_start, t_end) in enumerate(segments):
            try:
                # Extract features
                features = extract_features(seg_y, sr)
                if features is None or features.shape[0] != 162:
                    continue
                
                # Predict emotion
                x = prepare_for_model(features)
                if x is None:
                    continue
                    
                pred = model.predict(x, verbose=0) # type: ignore
                emo_idx = np.argmax(pred)
                emo_label = label_list[emo_idx] # type: ignore
                emo_conf = float(np.max(pred))
                
                # Transcribe
                transcript = transcribe_audio_data(seg_y, sr)
                
                results.append({
                    "segment": i+1,
                    "start_time": round(t_start, 2),
                    "end_time": round(t_end, 2),
                    "duration": round(t_end - t_start, 2),
                    "transcript": transcript,
                    "emotion": emo_label,
                    "confidence": round(emo_conf, 3)
                })
                
            except Exception as e:
                logger.error(f"Error processing segment {i}: {e}")
                continue

        # Calculate emotion statistics
        emo_counts = {}
        for r in results:
            emo_counts[r["emotion"]] = emo_counts.get(r["emotion"], 0) + 1

        logger.info(f"Successfully processed {len(results)} segments")
        
        return JSONResponse({
            "n_segments": len(results),
            "emotion_stats": emo_counts,
            "segments": results
        })

    except Exception as e:
        logger.error(f"Unexpected error in analyze_audio: {e}")
        return JSONResponse({
            "error": f"Internal server error: {str(e)}"
        }, status_code=500)
        
    finally:
        # Clean up temp directory
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)





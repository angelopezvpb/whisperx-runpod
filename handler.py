import os
import uuid
import gc
import requests
import runpod
import whisperx
import torch

# -------------------------------------------------------------------------
# ⚠️ PARCHE DE SEGURIDAD V2 (EL "NUCLEAR")
# PyTorch 2.6+ bloquea archivos de configuración de Pyannote.
# Aquí aplicamos AMBAS soluciones: Whitelist oficial y desactivación forzosa.
# -------------------------------------------------------------------------
try:
    # 1. SOLUCIÓN OFICIAL: Autorizar OmegaConf (usado por pyannote)
    # Esto soluciona el error "Unsupported global: GLOBAL omegaconf..."
    from omegaconf import DictConfig, ListConfig
    torch.serialization.add_safe_globals([DictConfig, ListConfig])
    print("[INFO] ✅ OmegaConf (ListConfig/DictConfig) añadido a la lista segura de PyTorch.")
except ImportError:
    print("[WARN] ⚠️ OmegaConf no encontrado. Si usas diarización, esto fallará.")
except Exception as e:
    print(f"[WARN] ⚠️ Error al añadir safe_globals: {e}")

# 2. SOLUCIÓN DE FUERZA BRUTA: Sobrescribir torch.load
# Por si alguna librería interna no usa el mecanismo estándar
original_load = torch.load
def safe_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = safe_load
print("[INFO] ✅ Parche global torch.load(weights_only=False) aplicado.")
# -------------------------------------------------------------------------


# ----------------------------
# Global caches
# ----------------------------
WHISPER_MODEL = None
WHISPER_DEVICE = None
WHISPER_COMPUTE_TYPE = None

ALIGN_CACHE = {}        # key: language_code -> (align_model, metadata)
DIARIZE_CACHE = {}      # key: (hf_token, device) -> diarize_pipeline


def _log_gpu_once():
    try:
        print(f"[env] torch={torch.__version__} cuda_available={torch.cuda.is_available()} cuda={torch.version.cuda}")
        if torch.cuda.is_available():
            print(f"[env] GPU: {torch.cuda.get_device_name(0)}  count={torch.cuda.device_count()}")
    except Exception as e:
        print(f"[env] gpu log failed: {e}")


_LOGGED = False


def _download_to_tmp(url: str) -> str:
    local_path = f"/tmp/audio_{uuid.uuid4().hex}"
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    return local_path


def _get_device_and_compute(input_data: dict):
    forced_device = input_data.get("device")
    if forced_device in ("cuda", "cpu"):
        device = forced_device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    compute_type = input_data.get("compute_type")
    if not compute_type:
        compute_type = "float16" if device == "cuda" else "int8"

    return device, compute_type


def _get_whisper_model(device: str, compute_type: str, language: str | None):
    global WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE

    if WHISPER_MODEL is None or WHISPER_DEVICE != device or WHISPER_COMPUTE_TYPE != compute_type:
        print(f"[model] loading whisper large-v3 device={device} compute_type={compute_type}")
        WHISPER_MODEL = whisperx.load_model(
            "large-v3",
            device,
            compute_type=compute_type,
            language=None 
        )
        WHISPER_DEVICE = device
        WHISPER_COMPUTE_TYPE = compute_type

    return WHISPER_MODEL


def _get_align(language_code: str, device: str):
    key = (language_code, device)
    if key not in ALIGN_CACHE:
        print(f"[align] loading align model lang={language_code} device={device}")
        align_model, metadata = whisperx.load_align_model(language_code=language_code, device=device)
        ALIGN_CACHE[key] = (align_model, metadata)
    return ALIGN_CACHE[key]


def _get_diarizer(hf_token: str, device: str):
    key = (hf_token, device)
    if key not in DIARIZE_CACHE:
        print(f"[diar] loading diarization pipeline device={device}")
        DIARIZE_CACHE[key] = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
    return DIARIZE_CACHE[key]


def handler(event):
    global _LOGGED
    if not _LOGGED:
        _log_gpu_once()
        _LOGGED = True

    try:
        input_data = event.get("input", {}) or {}

        audio_file = input_data.get("audio_file")
        if not audio_file:
            return {"error": "audio_file is required"}

        language = input_data.get("language")
        batch_size = int(input_data.get("batch_size", 16))

        align_output = bool(input_data.get("align_output", True))
        diarization = bool(input_data.get("diarization", False))
        min_speakers = input_data.get("min_speakers")
        max_speakers = input_data.get("max_speakers")

        device, compute_type = _get_device_and_compute(input_data)
        print(f"[job] device={device} compute_type={compute_type} batch_size={batch_size} diarization={diarization} align={align_output}")

        local_audio_path = audio_file
        if isinstance(audio_file, str) and audio_file.startswith(("http://", "https://")):
            local_audio_path = _download_to_tmp(audio_file)

        try:
            audio = whisperx.load_audio(local_audio_path)
        except Exception as e:
            return {"error": f"Failed to load audio: {str(e)}"}

        model = _get_whisper_model(device, compute_type, language)

        transcribe_kwargs = {"batch_size": batch_size}
        if language:
            transcribe_kwargs["language"] = language

        result = model.transcribe(audio, **transcribe_kwargs)

        if align_output:
            lang_code = result.get("language") or language
            if not lang_code:
                print("[align] skipped (no language detected)")
            else:
                try:
                    align_model, metadata = _get_align(lang_code, device)
                    result = whisperx.align(
                        result["segments"],
                        align_model,
                        metadata,
                        audio,
                        device,
                        return_char_alignments=False
                    )
                except Exception as e:
                    print(f"[align] error: {e}")
                    pass

        if diarization:
            hf_token = input_data.get("huggingface_access_token")
            if not hf_token:
                return {"error": "huggingface_access_token required for diarization"}

            try:
                diarizer = _get_diarizer(hf_token, device)
                diarize_segments = diarizer(
                    audio,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers
                )
                result = whisperx.assign_word_speakers(diarize_segments, result)
            except Exception as e:
                print(f"[diarization] error: {e}")
                if "weights_only" in str(e):
                     return {"error": f"Security Error: PyTorch blocked the model load. Details: {str(e)}"}
                return {"error": f"Diarization failed: {str(e)}"}

        try:
            if local_audio_path.startswith("/tmp/audio_") and os.path.exists(local_audio_path):
                os.remove(local_audio_path)
        except Exception:
            pass

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "segments": result.get("segments", []),
            "detected_language": result.get("language"),
            "language_probability": result.get("language_probability"),
        }

    except Exception as e:
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})

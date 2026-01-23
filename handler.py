import os
import uuid
import gc
import requests
import runpod
import whisperx
import torch

# -------------------------------------------------------------------------
# ✅ FIX PyTorch 2.6+ (weights_only=True por defecto) + OmegaConf allowlist
# -------------------------------------------------------------------------
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

def _allowlist_omegaconf_for_torch():
    try:
        if not hasattr(torch, "serialization") or not hasattr(torch.serialization, "add_safe_globals"):
            print("[INFO] torch.serialization.add_safe_globals no disponible en esta versión de torch.")
            return

        from omegaconf import DictConfig, ListConfig
        try:
            from omegaconf.base import ContainerMetadata
            allow = [DictConfig, ListConfig, ContainerMetadata]
        except Exception:
            allow = [DictConfig, ListConfig]

        torch.serialization.add_safe_globals(allow)
        print("[INFO] ✅ OmegaConf allowlisted en torch.serialization.add_safe_globals().")
    except ImportError:
        print("[WARN] ⚠️ OmegaConf no encontrado. Si activas diarización, puede fallar.")
    except Exception as e:
        print(f"[WARN] ⚠️ Error allowlisting OmegaConf: {e}")

_allowlist_omegaconf_for_torch()

_original_torch_load = torch.load

def _safe_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)

torch.load = _safe_torch_load
print("[INFO] ✅ Parche global torch.load(weights_only=False) aplicado.")
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
# ✅ FIX WhisperX moderno: DiarizationPipeline se importa desde whisperx.diarize
# -------------------------------------------------------------------------
def _get_diarization_pipeline_class():
    try:
        from whisperx.diarize import DiarizationPipeline
        return DiarizationPipeline
    except Exception:
        return getattr(whisperx, "DiarizationPipeline", None)

DiarizationPipelineCls = _get_diarization_pipeline_class()
# -------------------------------------------------------------------------


# ----------------------------
# Global caches
# ----------------------------
WHISPER_MODEL = None
WHISPER_DEVICE = None
WHISPER_COMPUTE_TYPE = None

ALIGN_CACHE = {}        # key: (language_code, device) -> (align_model, metadata)
DIARIZE_CACHE = {}      # key: (hf_token, device) -> diarize_pipeline

_LOGGED = False


def _log_gpu_once():
    try:
        print(f"[env] torch={torch.__version__} cuda_available={torch.cuda.is_available()} cuda={torch.version.cuda}")
        if torch.cuda.is_available():
            print(f"[env] GPU: {torch.cuda.get_device_name(0)}  count={torch.cuda.device_count()}")
    except Exception as e:
        print(f"[env] gpu log failed: {e}")


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


def _get_whisper_model(device: str, compute_type: str):
    global WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE

    if WHISPER_MODEL is None or WHISPER_DEVICE != device or WHISPER_COMPUTE_TYPE != compute_type:
        print(f"[model] loading whisper large-v3 device={device} compute_type={compute_type}")
        WHISPER_MODEL = whisperx.load_model(
            "large-v3",
            device,
            compute_type=compute_type,
            language=None  # autodetect
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
    if DiarizationPipelineCls is None:
        raise RuntimeError(
            "Tu instalación de whisperx no expone DiarizationPipeline. "
            "En WhisperX moderno debe existir en whisperx.diarize."
        )

    key = (hf_token, device)
    if key not in DIARIZE_CACHE:
        print(f"[diar] loading diarization pipeline device={device}")
        DIARIZE_CACHE[key] = DiarizationPipelineCls(use_auth_token=hf_token, device=device)
    return DIARIZE_CACHE[key]


def build_clips_from_segments(segments, gap=0.35, pad_start=0.10, pad_end=0.20):
    """
    Construye clips estables a partir de los timestamps de segmentos (Whisper).
    - gap: si la pausa entre segmentos supera esto (s), se corta clip.
    - pad_start / pad_end: margen para no cortar consonantes/respiraciones.
    """
    clips = []
    if not segments:
        return clips

    cur_start = float(segments[0]["start"])
    cur_end = float(segments[0]["end"])

    for s in segments[1:]:
        s_start = float(s["start"])
        s_end = float(s["end"])
        if (s_start - cur_end) <= gap:
            cur_end = max(cur_end, s_end)
        else:
            clips.append({
                "start": max(0.0, cur_start - pad_start),
                "end": max(0.0, cur_end + pad_end),
            })
            cur_start, cur_end = s_start, s_end

    clips.append({
        "start": max(0.0, cur_start - pad_start),
        "end": max(0.0, cur_end + pad_end),
    })
    return clips


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

        language = input_data.get("language")  # e.g. "es"
        batch_size = int(input_data.get("batch_size", 16))

        # IMPORTANTE: Para recorte fiable, por defecto alineado OFF.
        align_output = bool(input_data.get("align_output", False))
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

        model = _get_whisper_model(device, compute_type)

        # ✅ VAD + no contexto acumulado => timestamps más estables
        transcribe_kwargs = {
            "batch_size": batch_size,
            "vad_filter": True,
            "vad_parameters": {"min_silence_duration_ms": 300},
            "condition_on_previous_text": False,
        }
        if language:
            transcribe_kwargs["language"] = language

        result = model.transcribe(audio, **transcribe_kwargs)

        # ALIGN (opcional)
        if align_output:
            lang_code = result.get("language") or language
            if not lang_code:
                print("[align] skipped (no language detected)")
            else:
                try:
                    det_lang = result.get("language")
                    det_lang_prob = result.get("language_probability")

                    align_model, metadata = _get_align(lang_code, device)
                    aligned = whisperx.align(
                        result["segments"],
                        align_model,
                        metadata,
                        audio,
                        device,
                        return_char_alignments=False
                    )

                    # ✅ NO pisar 'result', solo actualizar lo necesario
                    result["segments"] = aligned.get("segments", result["segments"])
                    if "word_segments" in aligned:
                        result["word_segments"] = aligned["word_segments"]

                    result["language"] = det_lang
                    result["language_probability"] = det_lang_prob

                except Exception as e:
                    print(f"[align] error: {e}")

        # DIARIZATION (opcional)
        if diarization:
            hf_token = input_data.get("huggingface_access_token")
            if not hf_token:
                return {"error": "huggingface_access_token required for diarization"}

            try:
                diarizer = _get_diarizer(hf_token, device)

                diarize_kwargs = {}
                if min_speakers is not None:
                    diarize_kwargs["min_speakers"] = min_speakers
                if max_speakers is not None:
                    diarize_kwargs["max_speakers"] = max_speakers

                diarize_segments = diarizer(audio, **diarize_kwargs)
                result = whisperx.assign_word_speakers(diarize_segments, result)

            except Exception as e:
                print(f"[diarization] error: {e}")
                msg = str(e)
                if "Weights only load failed" in msg or "weights_only" in msg:
                    return {"error": f"Security Error: PyTorch blocked model load. Details: {msg}"}
                return {"error": f"Diarization failed: {msg}"}

        # ✅ Clips listos para recortar por ffmpeg (basados en segmentos)
        segments = result.get("segments", [])
        clips = build_clips_from_segments(
            segments,
            gap=float(input_data.get("clip_gap", 0.35)),
            pad_start=float(input_data.get("clip_pad_start", 0.10)),
            pad_end=float(input_data.get("clip_pad_end", 0.20)),
        )

        # Cleanup temp
        try:
            if isinstance(local_audio_path, str) and local_audio_path.startswith("/tmp/audio_") and os.path.exists(local_audio_path):
                os.remove(local_audio_path)
        except Exception:
            pass

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "segments": segments,
            "clips": clips,  # ✅ usa esto para recortar fiable
            "detected_language": result.get("language"),
            "language_probability": result.get("language_probability"),
        }

    except Exception as e:
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})



 

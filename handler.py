import os
import uuid
import gc
import requests
import runpod
import whisperx
import torch
import inspect
import re

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
# ✅ Compat HuggingFace Hub: mapear use_auth_token -> token si hace falta
# -------------------------------------------------------------------------
def _patch_hf_hub_download_compat():
    try:
        import huggingface_hub as hfh

        fn = getattr(hfh, "hf_hub_download", None)
        if fn is None:
            return

        try:
            sig = inspect.signature(fn)
            accepts_use_auth_token = "use_auth_token" in sig.parameters
            accepts_token = "token" in sig.parameters
        except Exception:
            accepts_use_auth_token = True
            accepts_token = True

        if accepts_use_auth_token or not accepts_token:
            return

        if getattr(fn, "_vp_use_auth_token_patched", False):
            return

        def _hf_hub_download_compat(*args, **kwargs):
            if "use_auth_token" in kwargs and "token" not in kwargs:
                kwargs["token"] = kwargs.pop("use_auth_token")
            else:
                kwargs.pop("use_auth_token", None)
            return fn(*args, **kwargs)

        _hf_hub_download_compat._vp_use_auth_token_patched = True
        hfh.hf_hub_download = _hf_hub_download_compat
        print("[INFO] ✅ Parche hf_hub_download(use_auth_token->token) aplicado.")
    except Exception as e:
        print(f"[WARN] ⚠️ No se pudo aplicar compat de huggingface_hub: {e}")


_patch_hf_hub_download_compat()
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


# -------------------------------------------------------------------------
# ✅ Compat helpers (kwargs "a prueba de versiones")
# -------------------------------------------------------------------------
def _filter_kwargs_by_signature(fn, kwargs: dict) -> dict:
    """Devuelve solo kwargs soportados por la firma de fn (evita unexpected keyword argument)."""
    try:
        sig = inspect.signature(fn)
        allowed = set(sig.parameters.keys())
        safe = {k: v for k, v in kwargs.items() if k in allowed}
        dropped = sorted(set(kwargs.keys()) - set(safe.keys()))
        if dropped:
            print(f"[compat] kwargs ignorados para {getattr(fn, '__name__', str(fn))}: {dropped}")
        return safe
    except Exception:
        # Si no podemos inspeccionar, no tocamos
        return kwargs

def _call_transcribe_safely(model, audio, **kwargs):
    safe_kwargs = _filter_kwargs_by_signature(model.transcribe, kwargs)
    print(f"[transcribe] kwargs iniciales: {sorted(list(safe_kwargs.keys()))}")
    retry_kwargs = dict(safe_kwargs)
    while True:
        try:
            return model.transcribe(audio, **retry_kwargs)
        except TypeError as e:
            msg = str(e)
            m = re.search(r"unexpected keyword argument '([^']+)'", msg)
            bad_kwarg = m.group(1) if m else None
            if bad_kwarg and bad_kwarg in retry_kwargs:
                print(f"[transcribe] retry sin kwarg incompatible: {bad_kwarg}")
                retry_kwargs.pop(bad_kwarg, None)
                print(f"[transcribe] kwargs retry: {sorted(list(retry_kwargs.keys()))}")
                continue
            print(f"[transcribe] TypeError sin fallback aplicable: {msg}")
            raise
# -------------------------------------------------------------------------


# ----------------------------
# Global caches
# ----------------------------
WHISPER_MODEL = None
WHISPER_DEVICE = None
WHISPER_COMPUTE_TYPE = None
WHISPER_LITERAL_MODE = None

ALIGN_CACHE = {}        # key: (language_code, device) -> (align_model, metadata)
DIARIZE_CACHE = {}      # key: (hf_token, device) -> diarize_pipeline

_LOGGED = False


def _parse_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return default


def _get_literal_mode(input_data: dict) -> bool:
    if "literal_mode" in input_data:
        return _parse_bool(input_data.get("literal_mode"), default=False)
    return _parse_bool(os.getenv("WHISPER_LITERAL_MODE"), default=False)


def _get_language(input_data: dict) -> str | None:
    language = input_data.get("language")
    if language is not None:
        language = str(language).strip().lower()
        return language or None

    env_language = os.getenv("WHISPER_LANGUAGE")
    if env_language is not None:
        env_language = env_language.strip().lower()
        return env_language or None

    return None


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


def _get_whisper_model(device: str, compute_type: str, language: str | None, literal_mode: bool):
    """
    Carga el modelo de WhisperX de forma compatible:
    - Intenta inyectar VAD/anti-deriva mediante asr_options si load_model lo soporta.
    - Si no, no rompe.
    """
    global WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE, WHISPER_LITERAL_MODE

    if (
        WHISPER_MODEL is None
        or WHISPER_DEVICE != device
        or WHISPER_COMPUTE_TYPE != compute_type
        or WHISPER_LITERAL_MODE != literal_mode
    ):
        print(
            f"[model] loading whisper large-v3 device={device} "
            f"compute_type={compute_type} literal_mode={literal_mode}"
        )

        # Preferencias para timestamps más estables (si están soportadas por tu build)
        asr_options = {
            "vad_filter": not literal_mode,
            "vad_parameters": {"min_silence_duration_ms": 300},
            "condition_on_previous_text": False,
        }

        load_kwargs = {
            "compute_type": compute_type,
            "language": None,  # autodetect
            "asr_options": asr_options,
        }

        safe_load_kwargs = _filter_kwargs_by_signature(whisperx.load_model, load_kwargs)

        WHISPER_MODEL = whisperx.load_model(
            "large-v3",
            device,
            **safe_load_kwargs
        )
        WHISPER_DEVICE = device
        WHISPER_COMPUTE_TYPE = compute_type
        WHISPER_LITERAL_MODE = literal_mode

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

        # Compat entre versiones: algunas esperan use_auth_token, otras token
        # y otras toman el token desde variables de entorno de HF.
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", hf_token)
        os.environ.setdefault("HF_TOKEN", hf_token)

        init_attempts = [
            {"use_auth_token": hf_token, "device": device},
            {"token": hf_token, "device": device},
            {"device": device},
        ]

        last_error = None
        for init_kwargs in init_attempts:
            try:
                DIARIZE_CACHE[key] = DiarizationPipelineCls(**init_kwargs)
                print(f"[diar] pipeline init ok with kwargs={sorted(list(init_kwargs.keys()))}")
                break
            except TypeError as e:
                last_error = e
                print(f"[diar] init retry por TypeError con kwargs={sorted(list(init_kwargs.keys()))}: {e}")
                continue
            except Exception as e:
                last_error = e
                print(f"[diar] init failed con kwargs={sorted(list(init_kwargs.keys()))}: {e}")
                continue

        if key not in DIARIZE_CACHE:
            raise RuntimeError(f"No se pudo inicializar DiarizationPipeline: {last_error}")
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


def _parse_audio_files(audio_file_value):
    if isinstance(audio_file_value, str):
        return [item.strip() for item in audio_file_value.split(";") if item.strip()]
    return []


def _process_single_audio(
    audio_file: str,
    input_data: dict,
    device: str,
    compute_type: str,
    language: str | None,
    batch_size: int,
    align_output: bool,
    diarization: bool,
    min_speakers,
    max_speakers,
    literal_mode: bool,
    item_index: int = 1,
    total_items: int = 1,
):
    print(f"[audio {item_index}/{total_items}] start source={audio_file}")
    local_audio_path = audio_file

    try:
        if audio_file.startswith(("http://", "https://")):
            print(f"[audio {item_index}/{total_items}] downloading remote audio")
            local_audio_path = _download_to_tmp(audio_file)
            print(f"[audio {item_index}/{total_items}] downloaded to {local_audio_path}")

        try:
            audio = whisperx.load_audio(local_audio_path)
            print(f"[audio {item_index}/{total_items}] audio loaded")
        except Exception as e:
            print(f"[audio {item_index}/{total_items}] load_audio failed: {e}")
            return {"error": f"Failed to load audio: {str(e)}"}

        model = _get_whisper_model(device, compute_type, language, literal_mode)
        print(f"[audio {item_index}/{total_items}] model ready")

        transcribe_kwargs = {
            "batch_size": batch_size,
        }
        if literal_mode:
            # Perfil más conservador para reducir normalizaciones y deriva.
            transcribe_kwargs["temperature"] = 0.0
        if language:
            transcribe_kwargs["language"] = language

        print(f"[audio {item_index}/{total_items}] transcribe start")
        result = _call_transcribe_safely(model, audio, **transcribe_kwargs)
        print(f"[audio {item_index}/{total_items}] transcribe done segments={len(result.get('segments', []))}")

        if align_output:
            print(f"[audio {item_index}/{total_items}] align enabled")
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

                    result["segments"] = aligned.get("segments", result["segments"])
                    if "word_segments" in aligned:
                        result["word_segments"] = aligned["word_segments"]

                    result["language"] = det_lang
                    result["language_probability"] = det_lang_prob
                    print(f"[audio {item_index}/{total_items}] align done")

                except Exception as e:
                    print(f"[align] error: {e}")

        if diarization:
            print(f"[audio {item_index}/{total_items}] diarization enabled")
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
                print(f"[audio {item_index}/{total_items}] diarization done")

            except Exception as e:
                print(f"[diarization] error: {e}")
                msg = str(e)
                if "Weights only load failed" in msg or "weights_only" in msg:
                    return {"error": f"Security Error: PyTorch blocked model load. Details: {msg}"}
                return {"error": f"Diarization failed: {msg}"}

        segments = result.get("segments", [])
        clips = build_clips_from_segments(
            segments,
            gap=float(input_data.get("clip_gap", 0.35)),
            pad_start=float(input_data.get("clip_pad_start", 0.10)),
            pad_end=float(input_data.get("clip_pad_end", 0.20)),
        )
        print(f"[audio {item_index}/{total_items}] clips built count={len(clips)}")

        return {
            "segments": segments,
            "clips": clips,
            "detected_language": result.get("language"),
            "language_probability": result.get("language_probability"),
            "literal_mode": literal_mode,
        }

    finally:
        try:
            if local_audio_path.startswith("/tmp/audio_") and os.path.exists(local_audio_path):
                os.remove(local_audio_path)
                print(f"[audio {item_index}/{total_items}] tmp removed")
        except Exception:
            pass

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[audio {item_index}/{total_items}] done")


def handler(event):
    global _LOGGED
    if not _LOGGED:
        _log_gpu_once()
        _LOGGED = True

    try:
        input_data = event.get("input", {}) or {}

        audio_files = _parse_audio_files(input_data.get("audio_file"))
        if not audio_files:
            return {"error": "audio_file is required"}
        print(f"[job] audio_files_count={len(audio_files)}")

        language = _get_language(input_data)  # e.g. "es"
        batch_size = int(input_data.get("batch_size", 16))
        literal_mode = _get_literal_mode(input_data)

        # Para recorte fiable, por defecto alineado OFF.
        align_output = bool(input_data.get("align_output", False))
        diarization = bool(input_data.get("diarization", False))

        min_speakers = input_data.get("min_speakers")
        max_speakers = input_data.get("max_speakers")

        device, compute_type = _get_device_and_compute(input_data)
        print(
            f"[job] device={device} compute_type={compute_type} batch_size={batch_size} "
            f"diarization={diarization} align={align_output} literal_mode={literal_mode}"
        )

        if len(audio_files) == 1:
            return _process_single_audio(
                audio_files[0],
                input_data,
                device,
                compute_type,
                language,
                batch_size,
                align_output,
                diarization,
                min_speakers,
                max_speakers,
                literal_mode,
                item_index=1,
                total_items=1,
            )

        results = []
        for index, audio_file in enumerate(audio_files, start=1):
            item_result = _process_single_audio(
                audio_file,
                input_data,
                device,
                compute_type,
                language,
                batch_size,
                align_output,
                diarization,
                min_speakers,
                max_speakers,
                literal_mode,
                item_index=index,
                total_items=len(audio_files),
            )
            item_result["audio_file"] = audio_file
            results.append(item_result)

        return results

    except Exception as e:
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})




 

# WhisperX Worker für Runpod

WhisperX Serverless Worker mit Speaker Diarization (pyannote.audio 3.1)

## Features
- Whisper large-v3
- Speaker Diarization
- Word-level timestamps
- Automatic language detection

## Input Parameters
```json
{
  "input": {
    "audio_file": "https://example.com/audio.mp3",
    "language": "de",
    "literal_mode": false,
    "diarization": true,
    "huggingface_access_token": "hf_...",
    "min_speakers": 2,
    "max_speakers": 5,
    "align_output": true,
    "batch_size": 16
  }
}
```

`audio_file` acepta uno o varios audios. Para varios, envialos en el mismo string separados por `;`:

```json
{
  "input": {
    "audio_file": "https://example.com/audio1.mp3;https://example.com/audio2.mp3"
  }
}
```

Si envias un solo audio, la respuesta mantiene el formato actual. Si envias varios, la respuesta sera un array con un resultado por cada audio, incluyendo su campo `audio_file` y errores por item si alguno falla.

`literal_mode` tambien se puede activar por variable de entorno del contenedor:

```dockerfile
ENV WHISPER_LITERAL_MODE=true
ENV WHISPER_LANGUAGE=es
```

Si mandas `input.literal_mode` o `input.language`, esos valores tienen prioridad sobre el `ENV`.

## Setup

1. Accept HuggingFace model access:
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0

2. Deploy to Runpod Serverless
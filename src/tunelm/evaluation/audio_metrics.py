"""Frozen audio scoring protocol for CLAP, FAD, FAD-CLAP and MAD."""

from __future__ import annotations

import hashlib
import importlib.metadata
import math
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from tunelm.core.io import content_hash, output_path, read_json
from tunelm.evaluation.audio_eval import AUDIO_SUFFIXES

MERT_MODEL = "m-a-p/MERT-v1-330M"
MERT_REVISION = "5240c2708a5acaee1007f43fb9735c7dcd0b78c9"
FMA_POP_SHA256 = "74746819873d7082498b0caec0c7846b2e7967050353761a3fdcfcabf3f11dcb"
CLAP_MODEL = "630k-audioset-best.pt"
SAMPLE_RATE = 48_000
MERT_LAYER = 24
MAUVE_SEED = 25


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def decode_audio(path: Path, seconds: float) -> np.ndarray:
    """Decode the exact scoring window from the first audio stream."""
    frames = round(seconds * SAMPLE_RATE)
    command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-protocol_whitelist",
        "file,pipe",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-t",
        str(seconds),
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-f",
        "s16le",
        "-",
    ]
    result = subprocess.run(command, capture_output=True, timeout=60, check=False)
    if result.returncode:
        message = result.stderr.decode(errors="replace").strip()
        raise ValueError(f"Cannot decode audio {path}: {message}")
    samples = np.frombuffer(result.stdout, dtype="<i2")
    if samples.size < frames:
        raise ValueError(f"{path} has {samples.size / SAMPLE_RATE:.3f}s; need {seconds:g}s")
    if not np.any(samples):
        raise ValueError(f"{path} is silent in the scoring window")
    return samples[:frames].astype(np.float32) / 32768.0


def _save_npz(path: Path, **arrays) -> None:
    destination = output_path(path)
    with NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            np.savez(handle, **arrays)
            handle.close()
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)


def _fad_reference(path: Path | None) -> tuple[Path, str, dict]:
    if path is None:
        dist = importlib.metadata.distribution("fadtk")
        path = Path(dist.locate_file("fadtk/stats/fma_pop.npz"))
        if not path.is_file() or file_sha256(path) != FMA_POP_SHA256:
            raise ValueError("Expected the pinned FADtk FMA-Pop statistics or --fad-reference")
    digest = file_sha256(path)
    with np.load(path, allow_pickle=False) as stats:
        references = {}
        for name, size in [("vggish", 128), ("clap-laion-audio", 512)]:
            mean, cov = stats[f"{name}.mu"], stats[f"{name}.cov"]
            if mean.shape != (size,) or cov.shape != (size, size):
                raise ValueError(f"Invalid {name} reference statistics in {path}")
            if not np.isfinite(mean).all() or not np.isfinite(cov).all():
                raise ValueError(f"Non-finite {name} reference statistics in {path}")
            references[name] = (mean, cov)
    return path, digest, references


class AudioEmbedders:
    """Load each scoring encoder once and reuse audio features across models."""

    def __init__(self, device):
        self.device = device
        self.vggish = None
        self.clap = None
        self.mert = None
        self.processor = None

    def _load_fad(self) -> None:
        from fadtk.model_loader import CLAPLaionModel, VGGishModel

        self.vggish = VGGishModel()
        self.clap = CLAPLaionModel("audio")
        for model in (self.vggish, self.clap):
            model.device = self.device
            print(f"Loading {model.name} encoder", file=sys.stderr, flush=True)
            with open(os.devnull, "w") as quiet, redirect_stdout(quiet):
                model.load_model()

    def _load_mert(self) -> None:
        from transformers import AutoModel, Wav2Vec2FeatureExtractor

        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            MERT_MODEL, revision=MERT_REVISION
        )
        self.mert = (
            AutoModel.from_pretrained(
                MERT_MODEL, revision=MERT_REVISION, trust_remote_code=True, use_safetensors=False
            )
            .to(self.device)
            .eval()
        )

    def mert_embedding(self, samples: np.ndarray) -> np.ndarray:
        import torch
        from scipy.signal import resample_poly

        if self.mert is None:
            self._load_mert()
        wav = resample_poly(samples, 1, 2)
        inputs = self.processor(wav, sampling_rate=24_000, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            hidden = self.mert(**inputs, output_hidden_states=True).hidden_states[MERT_LAYER]
        return hidden.amax(dim=1).squeeze(0).float().cpu().numpy()

    def fad_embeddings(self, samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        import torch
        from scipy.signal import resample_poly

        if self.clap is None:
            self._load_fad()
        with torch.inference_mode():
            vggish = self.vggish.get_embedding(resample_poly(samples, 1, 3))
            clap = self.clap.get_embedding(samples)
        return vggish, clap

    def text_embeddings(self, prompts: list[str]) -> dict[str, np.ndarray]:
        import torch

        if self.clap is None:
            self._load_fad()
        embeddings = {}
        for start in range(0, len(prompts), 32):
            batch = prompts[start : start + 32]
            with torch.inference_mode():
                vectors = self.clap.model.get_text_embedding(batch, use_tensor=True)
            embeddings.update(zip(batch, vectors.float().cpu().numpy(), strict=True))
        return embeddings


def _embed_reference(files, reference, encoders, seconds):
    features, excluded = [], []
    provenance = reference.parent / "source.json"
    fma = provenance.is_file() and read_json(provenance).get("archive", "").endswith(
        "/fma_large.zip"
    )
    for index, path in enumerate(files, 1):
        try:
            waveform = decode_audio(path, seconds)
        except (ValueError, subprocess.TimeoutExpired):
            if not fma:
                raise
            excluded.append(str(path.relative_to(reference)))
        else:
            features.append(encoders.mert_embedding(waveform))
        if index % 500 == 0 or index == len(files):
            print(f"MAD reference {index}/{len(files)} clips", file=sys.stderr, flush=True)
    if len(features) < 2 or len(excluded) > len(files) // 100:
        raise ValueError(f"MAD reference has {len(excluded)} unusable tracks out of {len(files)}")
    return np.stack(features), excluded


def _reference_mert(reference: Path, output: Path, encoders: AudioEmbedders, seconds: float):
    archive = reference / f"mad-{seconds:g}s.npz" if reference.is_dir() else reference
    if reference.is_dir() and not archive.is_file():
        files = sorted(
            file
            for file in reference.rglob("*")
            if file.is_file() and file.suffix.lower() in AUDIO_SUFFIXES
        )
        if len(files) < 2:
            raise ValueError(
                f"No saved {seconds:g}s MAD reference or usable reference audio in {reference}"
            )
        source_hash = content_hash(
            [(str(path.relative_to(reference)), file_sha256(path)) for path in files]
        )
        archive = output / f"mad-reference-{source_hash[:16]}-{seconds:g}s.npz"
        if not archive.exists():
            print(f"Embedding {len(files)} MAD reference clips", file=sys.stderr)
            embeddings, excluded = _embed_reference(files, reference, encoders, seconds)
            _save_npz(
                archive,
                embeddings=embeddings,
                seconds=seconds,
                revision=MERT_REVISION,
                source_sha256=source_hash,
                layer=MERT_LAYER,
                excluded=np.asarray(excluded, dtype=str),
            )
    with np.load(archive, allow_pickle=False) as data:
        if (str(data["revision"].item()), data["layer"].item(), data["seconds"].item()) != (
            MERT_REVISION,
            MERT_LAYER,
            seconds,
        ):
            raise ValueError("MAD reference must use the same MERT revision, layer and window")
        embeddings = data["embeddings"]
        excluded = data["excluded"].tolist() if "excluded" in data else []
    if (
        embeddings.ndim != 2
        or embeddings.shape[1] != 1024
        or len(embeddings) < 2
        or not np.isfinite(embeddings).all()
    ):
        raise ValueError(f"Invalid MERT reference features in {archive}")
    return embeddings, {
        "path": str(archive),
        "sha256": file_sha256(archive),
        "clips": len(embeddings),
        "excluded_tracks": excluded,
        "source": str(reference),
    }


def _clip_features(row, encoders: AudioEmbedders, cache: Path, seconds: float):
    artifact = cache / f"{file_sha256(row.audio_path)}.npz"
    if not artifact.exists():
        waveform = decode_audio(row.audio_path, seconds)
        vggish, clap = encoders.fad_embeddings(waveform)
        mert = encoders.mert_embedding(waveform)
        _save_npz(artifact, vggish=vggish, clap=clap, mert=mert)
    with np.load(artifact, allow_pickle=False) as data:
        return data["vggish"], data["clap"], data["mert"]


def _clap_score(audio: np.ndarray, text: np.ndarray) -> float:
    audio = audio.astype(np.float32)
    norms = np.maximum(np.linalg.norm(audio, axis=1) * np.linalg.norm(text), 1e-8)
    return float(np.clip(((audio @ text) / norms).mean(), -1, 1))


def _fad(features: list[np.ndarray], reference: tuple) -> float:
    from fadtk.fad import calc_frechet_distance

    points = np.concatenate(features).astype(np.float64)
    return float(
        calc_frechet_distance(points.mean(axis=0), np.cov(points, rowvar=False), *reference)
    )


def _mad(features: list[np.ndarray], reference: np.ndarray) -> float:
    import faiss
    from mauve import compute_mauve

    workers = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(1)
    try:
        result = compute_mauve(
            p_features=np.stack(features),
            q_features=reference,
            seed=MAUVE_SEED,
            verbose=False,
        )
    finally:
        faiss.omp_set_num_threads(workers)
    return -math.log(float(result.mauve))


def _score_model(rows, text, encoders, cache, seconds, fad_refs, mert_ref):
    vggish, clap, mert, details = [], [], [], []
    for row in rows:
        vgg, audio_clap, feat = _clip_features(row, encoders, cache, seconds)
        vggish.append(vgg)
        clap.append(audio_clap)
        mert.append(feat)
        details.append(
            {
                "model_name": row.model_name,
                "prompt_id": row.prompt_id,
                "audio_path": str(row.audio_path),
                "clap": _clap_score(audio_clap, text[row.prompt]),
            }
        )
    result = {
        "model_name": rows[0].model_name,
        "prompts": len(rows),
        "clap": float(np.mean([row["clap"] for row in details])),
        "fad": _fad(vggish, fad_refs["vggish"]),
        "fad_clap": _fad(clap, fad_refs["clap-laion-audio"]),
        "mad": _mad(mert, mert_ref),
    }
    return result, details


def score_models(
    rows,
    output: Path,
    *,
    seconds: float,
    device: str,
    fad_reference,
    mad_reference,
    embedding_cache: Path | None = None,
):
    import torch

    if not math.isfinite(seconds) or not 1 <= seconds <= 60:
        raise ValueError("--seconds must be between 1 and 60")
    if not shutil.which("ffmpeg"):
        raise RuntimeError("Install ffmpeg to evaluate saved audio")
    active_device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    )
    if active_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run on CPU or use a Vast.ai GPU host.")
    path, digest, fad_refs = _fad_reference(fad_reference)
    versions = {
        key: importlib.metadata.version(key)
        for key in (
            "fadtk",
            "laion-clap",
            "torch",
            "transformers",
            "scipy",
            "mauve-text",
            "faiss-cpu",
        )
    }
    protocol = {
        "version": 1,
        "seconds": seconds,
        "window": "prefix",
        "audio": "ffmpeg mono PCM16 at 48000 Hz",
        "clap_checkpoint": CLAP_MODEL,
        "mert_revision": MERT_REVISION,
        "mert_layer": MERT_LAYER,
        "mert_pooling": "max",
        "mauve_seed": MAUVE_SEED,
        "device": active_device.type,
        "versions": versions,
    }
    cache = (embedding_cache or output / "embeddings") / content_hash(protocol)[:16]
    encoders = AudioEmbedders(active_device)
    mert_ref, mad_info = _reference_mert(mad_reference, output, encoders, seconds)
    text = encoders.text_embeddings(sorted({row.prompt for row in rows}))
    by_model = defaultdict(list)
    for row in rows:
        by_model[row.model_name].append(row)
    models, clips = [], []
    for name, predictions in sorted(by_model.items()):
        print(f"Scoring {name} on {len(predictions)} prompts", file=sys.stderr, flush=True)
        model, details = _score_model(
            predictions, text, encoders, cache, seconds, fad_refs, mert_ref
        )
        models.append(model)
        clips.extend(details)
    return {
        "protocol": protocol,
        "fad_reference": {"path": str(path), "sha256": digest},
        "mad_reference": mad_info,
        "models": models,
    }, clips

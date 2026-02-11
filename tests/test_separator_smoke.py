import numpy as np
import pytest
from ml_collections import ConfigDict


torch = pytest.importorskip("torch")


class _ZeroModel(torch.nn.Module):
    def __init__(self, num_stems: int = 1):
        super().__init__()
        self.num_stems = num_stems

    def forward(self, x):
        # x: (batch, channels, time)
        b, c, t = x.shape
        if self.num_stems == 1:
            return torch.zeros((b, c, t), dtype=x.dtype, device=x.device)
        return torch.zeros((b, self.num_stems, c, t), dtype=x.dtype, device=x.device)


def _cfg(*, target_instrument, instruments, chunk_size=64, num_overlap=2, stereo=True):
    return ConfigDict(
        dict(
            model=dict(stereo=stereo, sample_rate=44100),
            training=dict(target_instrument=target_instrument, instruments=list(instruments)),
            inference=dict(chunk_size=int(chunk_size), num_overlap=int(num_overlap)),
        )
    )


def test_demix_track_target_instrument_returns_original_length():
    from mel_band_roformer_vocal.demix import demix_track

    cfg = _cfg(target_instrument="vocals", instruments=["vocals"], chunk_size=64, num_overlap=2, stereo=True)
    model = _ZeroModel(num_stems=1)
    mix = torch.randn(2, 200, dtype=torch.float32)

    out = demix_track(config=cfg, model=model, mix=mix, device=torch.device("cpu"), use_amp=False)
    assert set(out.keys()) == {"vocals"}
    assert out["vocals"].shape == (2, 200)
    assert np.allclose(out["vocals"], 0.0)


def test_demix_track_multiple_stems_maps_to_instruments():
    from mel_band_roformer_vocal.demix import demix_track

    cfg = _cfg(target_instrument=None, instruments=["vocals", "other"], chunk_size=64, num_overlap=2, stereo=True)
    model = _ZeroModel(num_stems=2)
    mix = torch.randn(2, 120, dtype=torch.float32)

    out = demix_track(config=cfg, model=model, mix=mix, device=torch.device("cpu"), use_amp=False)
    assert set(out.keys()) == {"vocals", "other"}
    assert out["vocals"].shape == (2, 120)
    assert out["other"].shape == (2, 120)


def test_separator_separate_audio_mono_roundtrip_shapes():
    from mel_band_roformer_vocal.separator import Separator

    sep = Separator.__new__(Separator)
    sep.config = _cfg(target_instrument="vocals", instruments=["vocals"], chunk_size=64, num_overlap=2, stereo=True)
    sep.device = torch.device("cpu")
    sep.use_amp = False
    sep.model = _ZeroModel(num_stems=1)

    audio_mono = np.random.randn(200).astype(np.float32)
    result = sep.separate_audio(audio_mono, sample_rate=44100, return_instrumental=True)

    assert result.vocals is not None
    assert result.instrumental is not None
    assert result.vocals.shape == (200,)
    assert result.instrumental.shape == (200,)
    assert np.allclose(result.vocals, 0.0)
    assert np.allclose(result.instrumental, audio_mono)


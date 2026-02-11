def test_top_level_import():
    import mel_band_roformer_vocal as pkg

    assert isinstance(pkg.__version__, str)


def test_default_config_loads():
    from mel_band_roformer_vocal.config import load_config

    cfg = load_config()
    assert cfg.training.target_instrument == "vocals"
    assert "vocals" in list(cfg.training.instruments)
    assert int(cfg.inference.chunk_size) > 0


def test_load_state_dict_filters_rotary_freqs(tmp_path):
    import torch
    from mel_band_roformer_vocal.modeling import load_state_dict

    class Dummy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rotary_embed = torch.nn.Module()
            self.rotary_embed.register_buffer("inv_freq", torch.zeros(2))

    model = Dummy()
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save(
        {
            "rotary_embed.inv_freq": torch.ones(2),
            "rotary_embed.freqs": torch.zeros(4),
        },
        ckpt_path,
    )

    load_state_dict(model, str(ckpt_path))
    assert torch.all(model.rotary_embed.inv_freq == 1)

def test_top_level_import():
    import mel_band_roformer_vocal as pkg

    assert isinstance(pkg.__version__, str)


def test_default_config_loads():
    from mel_band_roformer_vocal.config import load_config

    cfg = load_config()
    assert cfg.training.target_instrument == "vocals"
    assert "vocals" in list(cfg.training.instruments)
    assert int(cfg.inference.chunk_size) > 0

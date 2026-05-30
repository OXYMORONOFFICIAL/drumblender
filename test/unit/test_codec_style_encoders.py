import torch

from drumblender.upgrades.encoders import APCodecStyleAttentionEncoder
from drumblender.upgrades.encoders import APCodecStyleEncoder
from drumblender.upgrades.encoders import BSCodecStyleAttentionEncoder
from drumblender.upgrades.encoders import BSCodecStyleEncoder
from drumblender.upgrades.encoders import SpectroStreamStyleAttentionEncoder
from drumblender.upgrades.encoders import SpectroStreamStyleEncoder
from drumblender.upgrades.encoders import WavTokenizerStyleAttentionEncoder
from drumblender.upgrades.encoders import WavTokenizerStyleEncoder


def _assert_frame_encoder_shape(encoder):
    x = torch.randn(2, 1, 4096)
    lengths = torch.tensor([4096, 2048], dtype=torch.long)

    with torch.no_grad():
        y = encoder(x, lengths=lengths)

    expected_frames = int(encoder.output_lengths(torch.tensor([4096]))[0].item())
    assert y.shape == (2, expected_frames, 128)

    invalid_start = int(encoder.output_lengths(lengths)[1].item())
    invalid = y[1, invalid_start:]
    if invalid.numel() > 0:
        torch.testing.assert_close(invalid, torch.zeros_like(invalid))


def _assert_attention_encoder_shape(encoder):
    x = torch.randn(2, 1, 4096)
    lengths = torch.tensor([4096, 2048], dtype=torch.long)

    with torch.no_grad():
        y = encoder(x, lengths=lengths)

    assert y.shape == (2, 128)


def test_wavtokenizer_style_encoder_shapes():
    _assert_frame_encoder_shape(
        WavTokenizerStyleEncoder(
            hidden_channels=4,
            latent_channels=16,
            output_channels=128,
            strides=(2, 2, 4, 8),
            lstm_layers=1,
            target_hop_size=128,
        )
    )
    _assert_attention_encoder_shape(
        WavTokenizerStyleAttentionEncoder(
            hidden_channels=4,
            latent_channels=16,
            output_channels=128,
            strides=(2, 2, 4, 8),
            lstm_layers=1,
            target_hop_size=128,
        )
    )


def test_bscodec_style_encoder_shapes():
    bands = ((0.0, 4000.0), (4000.0, 24000.0))
    _assert_frame_encoder_shape(
        BSCodecStyleEncoder(
            hidden_channels=4,
            latent_channels=16,
            output_channels=128,
            bands=bands,
            strides=(2, 2, 4, 8),
            lstm_layers=0,
            target_hop_size=128,
        )
    )
    _assert_attention_encoder_shape(
        BSCodecStyleAttentionEncoder(
            hidden_channels=4,
            latent_channels=16,
            output_channels=128,
            bands=bands,
            strides=(2, 2, 4, 8),
            lstm_layers=0,
            target_hop_size=128,
        )
    )


def test_apcodec_style_encoder_shapes():
    kwargs = dict(
        hidden_channels=8,
        branch_channels=16,
        latent_channels=8,
        output_channels=128,
        num_layers=1,
        n_fft=128,
        hop_size=16,
        win_size=64,
        spectral_downsample_ratio=4,
        target_hop_size=128,
    )
    _assert_frame_encoder_shape(APCodecStyleEncoder(**kwargs))
    _assert_attention_encoder_shape(APCodecStyleAttentionEncoder(**kwargs))


def test_spectrostream_style_encoder_shapes():
    kwargs = dict(
        hidden_channels=16,
        output_channels=128,
        n_ffts=(128, 256),
        num_layers=1,
        target_hop_size=128,
    )
    _assert_frame_encoder_shape(SpectroStreamStyleEncoder(**kwargs))
    _assert_attention_encoder_shape(SpectroStreamStyleAttentionEncoder(**kwargs))

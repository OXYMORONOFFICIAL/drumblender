import torch

from drumblender.upgrades.encoders import DACStyleEncoder
from drumblender.upgrades.encoders import DACStyleSequenceEncoder


def test_dac_style_sequence_encoder_preserves_noise_shape():
    x = torch.randn(2, 1, 4096)

    base = DACStyleEncoder(
        input_channels=1,
        hidden_channels=16,
        output_channels=128,
        strides=(2, 2, 4, 8),
        transpose_output=True,
    )
    upgraded = DACStyleSequenceEncoder(
        input_channels=1,
        hidden_channels=16,
        output_channels=128,
        strides=(2, 2, 4, 8),
        transpose_output=True,
        lstm_hidden_size=256,
        lstm_layers=2,
    )

    with torch.no_grad():
        y_base = base(x)
        y_upgraded = upgraded(x)

    assert y_upgraded.shape == y_base.shape
    assert y_upgraded.shape[-1] == 128


def test_dac_output_lengths_match_runtime_shape():
    encoder = DACStyleEncoder(
        input_channels=1,
        hidden_channels=16,
        output_channels=128,
        strides=(2, 2, 4, 8),
        transpose_output=True,
    )

    for length in (128, 129, 191, 192, 256, 257, 4096, 48000):
        x = torch.randn(1, 1, length)
        with torch.no_grad():
            y = encoder(x)
        expected = int(encoder.output_lengths(torch.tensor([length]))[0].item())
        assert y.shape[1] == expected


def test_dac_len_zero_masks_invalid_frames():
    x = torch.randn(2, 1, 4096)
    lengths = torch.tensor([4096, 2048], dtype=torch.long)
    x[1, :, 2048:] = torch.randn_like(x[1, :, 2048:])

    encoder = DACStyleEncoder(
        input_channels=1,
        hidden_channels=16,
        output_channels=128,
        strides=(2, 2, 4, 8),
        transpose_output=True,
        padding_aware_backbone=True,
    )

    with torch.no_grad():
        y = encoder(x, lengths=lengths)

    frame_lengths = encoder.output_lengths(lengths)
    invalid = y[1, frame_lengths[1] :]
    assert invalid.numel() > 0
    torch.testing.assert_close(invalid, torch.zeros_like(invalid))


def test_dac_lstm_len_zero_masks_invalid_frames():
    x = torch.randn(2, 1, 4096)
    lengths = torch.tensor([4096, 2048], dtype=torch.long)
    x[1, :, 2048:] = torch.randn_like(x[1, :, 2048:])

    encoder = DACStyleSequenceEncoder(
        input_channels=1,
        hidden_channels=16,
        output_channels=128,
        strides=(2, 2, 4, 8),
        transpose_output=True,
        lstm_hidden_size=256,
        lstm_layers=2,
        length_aware_lstm=True,
        mask_invalid_frames=True,
    )

    with torch.no_grad():
        y = encoder(x, lengths=lengths)

    frame_lengths = encoder.encoder.output_lengths(lengths)
    invalid = y[1, frame_lengths[1] :]
    assert invalid.numel() > 0
    torch.testing.assert_close(invalid, torch.zeros_like(invalid))

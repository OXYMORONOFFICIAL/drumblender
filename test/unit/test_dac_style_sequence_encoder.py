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

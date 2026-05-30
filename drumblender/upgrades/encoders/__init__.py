"""
Experimental encoder modules.

Keep encoder variants isolated from baseline model code.
"""

from drumblender.upgrades.encoders.dac_style import DACStyleAttentionEncoder
from drumblender.upgrades.encoders.dac_style import DACStyleEncoder
from drumblender.upgrades.encoders.dac_style import DACStyleSequenceEncoder
from drumblender.upgrades.encoders.codec_styles import APCodecStyleAttentionEncoder
from drumblender.upgrades.encoders.codec_styles import APCodecStyleEncoder
from drumblender.upgrades.encoders.codec_styles import BSCodecStyleAttentionEncoder
from drumblender.upgrades.encoders.codec_styles import BSCodecStyleEncoder
from drumblender.upgrades.encoders.codec_styles import SpectroStreamStyleAttentionEncoder
from drumblender.upgrades.encoders.codec_styles import SpectroStreamStyleEncoder
from drumblender.upgrades.encoders.codec_styles import WavTokenizerStyleAttentionEncoder
from drumblender.upgrades.encoders.codec_styles import WavTokenizerStyleEncoder

__all__ = [
    "DACStyleEncoder",
    "DACStyleAttentionEncoder",
    "DACStyleSequenceEncoder",
    "WavTokenizerStyleEncoder",
    "WavTokenizerStyleAttentionEncoder",
    "BSCodecStyleEncoder",
    "BSCodecStyleAttentionEncoder",
    "APCodecStyleEncoder",
    "APCodecStyleAttentionEncoder",
    "SpectroStreamStyleEncoder",
    "SpectroStreamStyleAttentionEncoder",
]

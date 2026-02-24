"""
Experimental encoder modules.

Keep encoder variants isolated from baseline model code.
"""

from drumblender.upgrades.encoders.dac_style import DACStyleAttentionEncoder
from drumblender.upgrades.encoders.dac_style import DACStyleEncoder
from drumblender.upgrades.encoders.discodec_style import DisCodecStyleAttentionEncoder
from drumblender.upgrades.encoders.discodec_style import DisCodecStyleEncoder
from drumblender.upgrades.encoders.hybrid import HybridAttentionEncoder
from drumblender.upgrades.encoders.hybrid import HybridFrameEncoder

__all__ = [
    "DACStyleEncoder",
    "DACStyleAttentionEncoder",
    "DisCodecStyleEncoder",
    "DisCodecStyleAttentionEncoder",
    "HybridFrameEncoder",
    "HybridAttentionEncoder",
]

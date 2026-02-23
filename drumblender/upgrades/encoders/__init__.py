"""
Experimental encoder modules.

Keep encoder variants isolated from baseline model code.
"""

from drumblender.upgrades.encoders.dac_style import DACStyleAttentionEncoder
from drumblender.upgrades.encoders.dac_style import DACStyleEncoder

__all__ = ["DACStyleEncoder", "DACStyleAttentionEncoder"]

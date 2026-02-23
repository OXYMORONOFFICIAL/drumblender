"""Experimental loss modules.

Place optional training-loss upgrades here and select them via config toggles.
"""

from drumblender.upgrades.losses.safe_mss import SafeScaleInvariantMSSLoss

__all__ = ["SafeScaleInvariantMSSLoss"]

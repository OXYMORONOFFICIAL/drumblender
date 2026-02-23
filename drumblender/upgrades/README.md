# Upgrades Policy

This folder is for optional research upgrades only.

## Goals
- Preserve baseline reproducibility.
- Make every new feature easy to enable/disable.
- Keep experimental code separate from stable code.

## Folder layout
- `losses/`: training loss upgrades
- `encoders/`: encoder architecture upgrades
- Add new categories as needed (`synths/`, `metrics/`, etc.)

## Toggle strategy
- Add explicit config flags for each feature.
- Default must always be baseline-compatible.
- Do not change baseline config behavior silently.

## Recommended workflow
1. Implement upgrade in `drumblender/upgrades/<category>/`.
2. Add a dedicated config (`cfg/...`) for ON mode.
3. Keep baseline config untouched.
4. Compare ON/OFF with identical train script except toggle.


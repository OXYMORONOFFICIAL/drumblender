# Development TODO

This file tracks repo/development items that came up during the DrumBlender
encoder work. 

## Reference

- Modal feature entry points:
  - Legacy path: `drumblender/utils/modal_analysis.py` with
    `scripts/build_modal_features.py`.
  - New isolated path: `drumblender/utils/modal_analysis_NEW.py` with
    `scripts/build_modal_features_new.py`.

- Modal feature environment notes:
  - `modal_analysis_NEW.py` depends on `scipy.ndimage.median_filter`.
  - Modal feature extraction environments therefore need `scipy` in addition
    to the existing `torch` / `torchaudio` / `nnAudio` stack.

- Existing noise encoder modes:
  - `baseline`: SoundStream noise encoder.
  - `dac`: DAC-style noise encoder.
  - `dac_lstm`: DAC-style noise encoder with a 2-layer unidirectional LSTM tail.
  - `dac_len`: DAC-style noise encoder with padding-aware DAC backbone.
  - `dac_lstm_len`: plain DAC backbone plus length-aware LSTM tail.
  - `dac_len_lstm_len`: padding-aware DAC backbone plus length-aware LSTM tail.
  - `wavtokenizer`: WavTokenizer-style compact encoder.
  - `bscodec`: BSCodec-style compact encoder.
  - `apcodec`: APCodec-style compact encoder.
  - `spectrostream`: SpectroStream-style encoder.

- Existing transient encoder modes:
  - `baseline`: SoundStream attention encoder.
  - `dac`: DAC-style attention encoder.
  - `wavtokenizer`: WavTokenizer-style compact attention encoder.
  - `bscodec`: BSCodec-style compact attention encoder.
  - `apcodec`: APCodec-style compact attention encoder.
  - `spectrostream`: SpectroStream-style attention encoder.

- Completed development context:
  - DisCodec code/config/script references were removed.
  - Plain DAC encoder behavior was checked against the old DisCodec-era DAC
    implementation; plain noise DAC and transient DAC matched exactly.
  - `run.sh`, `test_all.sh`, and `test_per_pack.sh` support the current encoder
    mode strings.

## Codec-style encoder compact settings

All codec-style encoder variants must preserve the DrumBlender audio contract:

- Input audio stays 48 kHz mono waveform: `[B, 1, T]`.
- Noise encoder output stays `[B, ceil(T / 128), 128]`.
- Transient encoder output stays `[B, 128]`.
- Frame hop stays 128 samples, i.e. 375 fps at 48 kHz.
- Do not reduce the observed sample rate or output temporal resolution just to
  make the models smaller.

### WavTokenizer-style compact setting

- Original method meaning:
  - WavTokenizer is an efficient acoustic discrete codec/tokenizer for audio
    language modeling.
  - Official public configs are 24 kHz and target low token rates such as 40 or
    75 tokens/sec, using an EnCodec/SEANet-style encoder, RVQ, and a Vocos-like
    decoder.
  - In DrumBlender we only use the pre-RVQ waveform encoder idea, not the
    quantizer or decoder, and we keep the much denser 48 kHz / 128-sample-hop
    frame output.
- Full-style reference setting:
  - `hidden_channels=32`, `latent_channels=512`, `lstm_layers=2`.
  - Encoder pair params: about 14.31M.
- Selected compact setting:
  - `hidden_channels=32`, `latent_channels=512`, `lstm_layers=1`.
  - Encoder pair params: about 10.11M.
- What was reduced:
  - Kept the official-small-style SEANet width and 512-dim latent width.
  - Kept an LSTM tail, but reduced recurrent depth from 2 layers to 1.
- Expected effect:
  - Keeps the SEANet + recurrent-context flavor.
  - Smaller than the full WavTokenizer-style encoder without changing what the
    model observes.
  - Less long-context capacity than full style, but still not an aggressive
    low-capacity rewrite.

### BSCodec-style compact setting

- Original method meaning:
  - BSCodec is a band-split neural audio codec.
  - Its main idea is to split audio into frequency bands and encode each band
    separately, so low/mid/high regions can receive separate modeling capacity.
  - Official recipes include 2-band, 3-band, and 5-band variants at 24 kHz.
- Full-style reference setting:
  - 48 kHz mono, 3 bands covering `0-2k`, `2-8k`, `8-24k`.
  - Per-band SEANet-style encoders with `hidden_channels=32`,
    `latent_channels=128`, `lstm_layers=1`.
  - Encoder pair params: about 21.64M.
- Selected compact setting:
  - 2 bands, `hidden_channels=32`, `latent_channels=128`, `lstm_layers=1`.
  - Encoder pair params: about 14.45M.
- What was reduced:
  - Reduced band count from 3 to 2, following the official smaller BSCodec
    recipe direction.
  - Kept each per-band encoder width and latent width unchanged.
  - Kept the LSTM tail.
- Expected effect / caveat:
  - Still covers the full 0-24 kHz range at 48 kHz.
  - Still tests the BSCodec band-split hypothesis.
  - Frequency partition resolution is lower than the 3-band version, so this is
    a stronger tradeoff than WavTokenizer/APCodec compact settings. Compare carefully,
    especially on samples where separating low body, mid attack, and high noise
    matters.

### APCodec-style compact setting

- Original method meaning:
  - APCodec is designed for 48 kHz audio and encodes amplitude and phase spectra
    in parallel.
  - Official code/config uses STFT features with `n_fft=1024`, `hop_size=40`,
    `win_size=320`, ConvNeXt-style amplitude/phase branches, and RVQ.
  - In DrumBlender we keep the amplitude/phase spectral encoder idea and remove
    RVQ/decoder.
- Full-style reference setting:
  - `hidden_channels=256`, `branch_channels=256`, `num_layers=8`,
    `latent_channels=32`.
  - Encoder pair params: about 13.58M.
- Selected compact setting:
  - `hidden_channels=192`, `branch_channels=192`, `num_layers=6`,
    `latent_channels=32`.
  - Encoder pair params: about 7.20M.
- What was reduced:
  - Moderately reduced ConvNeXt branch width.
  - Moderately reduced branch depth from 8 blocks to 6.
  - Kept `latent_channels=32`.
  - Keep STFT settings and output hop unchanged.
- Expected effect:
  - Preserves APCodec's core amplitude/phase split at 48 kHz.
  - Reduces branch capacity without lowering spectral analysis resolution.
  - This is the safest compact variant among the three because the original method
    is already a 48 kHz spectral codec and the lite change mostly trims network
    capacity rather than changing what the model observes.

Reference links:

- WavTokenizer paper/code: https://arxiv.org/abs/2408.16532,
  https://github.com/jishengpeng/WavTokenizer
- BSCodec code branch: https://github.com/whr-a/espnet/tree/bscodec
- APCodec paper/code: https://arxiv.org/abs/2402.10533,
  https://github.com/YangAi520/APCodec


## CQT 개선 관련 자료조사
synchrosqueezed ridge analysis
FRI 기반 grid-free ridge estimation
EDSM/eaQHM model을 refiner로 사용

최신 방법이 보는 난제는 grid dependence, ridge jitter, short fake ridge, curved IF estimation failure, high noise/signal density에서의 mode 간 간섭 등이 있다. 


의미 있는 물리 mode가 아니라, 샘플 곳곳에 흩어진 지역적 에너지 maxima가 채택되는 점이 본질적 문제. FRI 논문도 제안법이 좋은 성능을 보였지만, spatial constraint를 쓰지 않으면 작은 oscillation이 남는다고 스스로 적고 있다. 이건 매우 중요한 힌트다. 즉, 개선의 핵심은 단순히 “더 좋은 transform” 하나로 끝나지 않고, 연속성/곡률/수명(duration)/에너지 통합량을 넣은 ridge selection objective가 반드시 같이 가야 한다는 뜻이다. 

### Synchrosqueezed ridge analysis - GPT가 본 best
CQT를 버리고, forward-time의 synchrosqueezed CWT(Continuous Wavelet Transform) 또는 synchrosqueezed STFT로 ridge 뽑기. 
기본적으로 AM-FM component의 instantaneous frequency ridge를 더 날카롭게 모으기 위해 나온 방법.
단순 spectrogram/CQT local maxima보다 곡선형 frequency movement에 훨씬 잘 맞는다. 
SCT(Synchrosqueezed Chirplet Transform): 2021년 chirplet transform이 chirp-rate 축에서 blur된다는 문제를 지적하고, SCT가 crossover IF에서도 더 높은 contrast와 이론적 보장을 준다고 말한다. 
2024년 high-order SCT에서 나아가 기존 SCT가 high chirp modulation에서 IF를 잘못 추정하는 문제를 고친다고 주장한다. 

ssqueezepy는 Python에서 CWT/STFT synchrosqueezing, ridge extraction, generalized Morse wavelets, GPU/병렬 가속을 제공해서 적용 쉽다. generalized Morse wavelet은 2022년 음악 신호 연구에서 Morlet/Gabor보다 더 analytic하고, 비정상(nonstationary) oscillatory music signal에서 amplitude/phase/frequency 해석 가능성을 높인다고 설명. 
percussive에서 HF가 복잡하고 scale-dependent인 경우 CQT의 고정 log-bin 체계보다 analytic wavelet 기반 ridge가 더 괜찮을지도. 
다만 raw ridge point 그대로가 아닌 전체 경로 최적화 필요. 누적 에너지, 수명, 곡률 penalty, 주파수 1차/2차 smoothness 등..
2024년 noisy scalogram ridge 논문이 ridge의 uniqueness와 noisy-clean deviation bound를 다루는 것도, ridge를 “그때그때의 peak”가 아니라 “연속적 object”로 다뤄야 한다는 쪽을 지지한다.

### FRI 기반 grid-free ridge estimation
FRI-TLS / FRI-SST
2022년 spectrogram column을 Dirac pulse stream이 TF kernel로 blur된 관측으로 보고, ridge 위치를 finite rate of innovation reconstruction으로 복원. 
최종 IF 추정치가 TF grid resolution에 의존하지 않는다 -> 떨림/여기저기 짧은/계단식 freq drop 해결
SOTA ridge detector + pseudo-Bayesian 방법과 비교해 성능 개선. Code Ocean 코드도 제공한

spatial constraint가 없어서 추정치가 true IF 주변에서 작게 oscillate -> track continuity penalty를 넣지 않으면 여전히 떨림. 따라서 synchrosqueezed TFR 위의 어려운 구간이나 HF dense region에서 grid-free local refiner로 쓰자

### EDSM / eaQHM
ridge 찾고 나서 refine하는 용도로 쓰기. 2024년 비교 논문에서 EDSM을 subspace 기반 exponentially damped sinusoid model, eaQHM을 local characteristics에 적응하는 non-parametric time-varying basis로 설명. 
eaQHM은 medium-to-large window에서 EDSM보다 낫고, 반대로 EDSM은 작은 window에서 더 강하다는 결과. 또 EDSM은 damping factor를 직접 모델링하므로, strike 이후 decay를 갖는 퍼커시브 공통 구조와 잘 맞는다고 함. 반면 eaQHM은 frequency curve 자체에 basis를 적응시키므로, pitch/frequency drop이 있는 smooth curved track를 정련하는 데 더 적합. 
기본적으로 eaQHM은 parameter-refinement mechanism이라 초기 instantaneous component 추정에 의존 + small window에서는 LS conditioning 문제. 반대로 EDSM은 작은 window에서 잘 버티지만, window 안에서는 frequency stationarity를 가정. 
따라서 곡선형 이동을 강하게 보이는 드럼 모달에는 detector로 쓰기엔 한계가 있음. 



synchrosqueezed CWT with generalized Morse wavelets를 기본 time-frequency front-end로 둔 다음 그 위에서 전체 샘플 구간에 대해 ridge candidate를 만들고, Viterbi류 또는 연속 최적화 기반의 path objective로 64개 내외의 track를 선택. 
objective는 최소한 적분 에너지, 최소 지속시간, slope/curvature penalty, crossing penalty, HF 가중 threshold 포함해야 한다. 
여기에 dense/high-frequency region에서만 FRI-TLS 혹은 FRI-SST를 써서 ridge frequency를 grid에서 떼내기
마지막으로 각 track마다 짧은 window EDSM으로 damping과 local sinusoid parameter를 정련하기. 
smoother가 더 필요하면 probabilistic chirp smoother 같은 per-track refiner를 마지막 단계에 붙일 수 있음


실제 우선순위를 하나로 정리하면 이렇다.
최우선은 SSQ-CWT + continuity-regularized ridge tracking이다.
두 번째는 FRI를 HF/dense band의 grid-free local refiner로 추가하는 것이다.
세 번째는 EDSM을 track-level parametric refiner로 붙이는 것이다.
eaQHM은 좋은 방법이지만, current problem인 “검출 단계의 가짜 ridge”를 해결하는 1순위가 아니라서 뒤로 미룬다. eaQHM은 ridge가 이미 어느 정도 맞게 잡힌 뒤, 저중역의 더 smooth한 curved track를 예쁘게 다듬는 데 더 어울린다. 

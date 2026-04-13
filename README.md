## Training
Use the `drumblender` command to train a new model.

```bash
drumblender fit -c cfg/01_noise_params.yaml --data cfg/data/freesound.yaml
```

Training config files can be found in the directory `cfg`. The configuration files
in the root of that directory are the configurations used to train and test the different
model configurations presented in the Forum Acusticum paper.

We used the PyTorch Lightning
[LightningCLI](https://lightning.ai/docs/pytorch/LTS/api/pytorch_lightning.cli.LightningCLI.html?highlight=lightningcli#pytorch_lightning.cli.LightningCLI).

## Testing
Pass `test` as an argument to `drumblender` to test a trained model. For example, to test a model on the test set of the Freesound Percussive One-Shot dataset:

```bash
drumblender test -c models/forum-acusticum-2023/noise_parallel_transient_params.yaml --ckpt models/forum-acusticum-2023/noise_parallel_transient_params.ckpt --data cfg/data/freesound.yaml --trainer.logger CSVLogger --model.test_metrics cfg/metrics/drumblender_metrics.yaml
```

The `--trainer.logger` argument overrides the logging configuration in the saved yaml file. `--model.test_metrics` adds extra metrics used in the evaluation for the paper.

To run this command on a CPU you can add the argument `--trainer.accelerator cpu`

## GitHub Pages
This repository includes a static experiment board under `docs/` for publishing
evaluation summaries and a small curated audio audition set on GitHub Pages.

To rebuild the site from local evaluation outputs in `logs/`:

```bash
python scripts/build_github_pages.py --logs-root logs --output docs --samples-per-run 6
```

The script scans `logs/**/evaluation/summary.json`, copies a few
target/reconstruction audio pairs for runs that include `manifest.csv`, and
writes the site data to `docs/data/site-data.json`. A GitHub Actions workflow at
`.github/workflows/pages.yml` then deploys the `docs/` folder to GitHub Pages on
pushes to `main`.

## System Information
For Forum Acusticum 2023, experiments were run using

- Python version: 3.10.10.
- GPU: Tesla V100-PCIE-16GB

The exact Python packages in the environment during model training are in
`train-packages.txt`.

## For Developers
To install dev requirements and pre-commit hooks:

```bash
$ pip install -e ".[dev]"
```

Install pre-commit hooks if developing and contributing:

```bash
$ pre-commit install
```

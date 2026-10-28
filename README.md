# TI-GAE for flood type classification in the Yiluo River network

Processed inputs and TI-GAE code accompanying *Topology-informed graph learning for flood type classification and rainfall-based recognition in river networks*.

## Contents

- `data/flood_features.xlsx`: 42 worksheets (`1`–`42`) in event order. Each worksheet contains 11 stations, two identifier columns and 15 flood indicators.
- `data/events.csv`: event indices, original event identifiers, display identifiers and event time windows. Event identifiers are labels rather than observation dates.
- `data/stations.csv`: node indices, station names, abbreviations and longitude/latitude coordinates in WGS84.
- `data/topology.csv`: ten directed edges, from upstream to downstream, using zero-based node indices.
- `ti_gae_core.py`: graph attention encoder, graph pooling, decoder, reconstruction loss and silhouette-based selection of the number of clusters.
- `train_ti_gae.py`: input loading, normalization, training, clustering and indicator-subset search.
- `config.json`: input paths, hyperparameters, random seeds and selected indicators.

## Installation

Use Python 3.9 or later. From the repository directory:

```shell
python -m pip install -r requirements.txt
```

## Run

Check the input files:

```shell
python train_ti_gae.py --check-inputs
```

Train the selected five-indicator model with seed 42:

```shell
python train_ti_gae.py --seeds 42
```

Omitting `--seeds` runs seeds 42–46. The default model uses `total_volume`, `max_volume_12h`, `max_volume_24h`, `max_volume_48h` and `recession_v`.

Search all 32,647 subsets containing 3–15 indicators:

```shell
python train_ti_gae.py --all-subsets --seeds 42 43 44 45 46
```

The full search is computationally intensive. A search can be divided using zero-based `--start` and `--limit`, for example:

```shell
python train_ti_gae.py --all-subsets --seeds 42 --start 0 --limit 100 --output outputs_part1
```

Use `--device cuda` for a compatible GPU, `--epochs` to change the training duration, or `--output` to choose an output directory. Existing per-model output directories are protected against overwriting.

## Input conventions

Station order is fixed across all worksheets and matches `data/stations.csv`. Flood volumes are expressed in million cubic metres; discharge is in cubic metres per second; time is in hours. The feature fields are:

`duration_hours`, `total_volume`, `max_volume_3h`, `max_volume_12h`, `max_volume_24h`, `max_volume_48h`, `concentration`, `peak_flow`, `Cs`, `FT`, `count_above_75_percent`, `count_above_50_percent`, `Cv`, `Rise_v`, `recession_v`.

An all-zero selected feature vector gives a flood-presence flag of zero. Per-station normalization uses valid vectors across the 42 events. The first 40 events are used for reconstruction training and the last two for reconstruction validation. Continuous reconstruction loss is masked by the flood-presence flag; flag reconstruction uses binary cross-entropy. Graph embeddings from all 42 events are clustered.

The encoder has two graph attention layers (32 hidden units), mean/max/sum pooling and an 8-dimensional graph embedding. Training uses Adam, learning rate 0.001 and 300 epochs. Candidate cluster counts are 2–10, selected by silhouette score.

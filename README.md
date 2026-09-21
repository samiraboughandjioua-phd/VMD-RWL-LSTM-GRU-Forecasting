# VMD-RWL-LSTM-GRU-Forecasting

Reproducible code accompanying the manuscript on multi-horizon hourly river water-level (RWL) forecasting using LSTM and GRU models at Vam Nao, Tan Chau, and Can Tho in the Vietnamese Mekong Delta.

## Repository contents

* `01\_Model\_Training/` — LSTM and GRU training scripts.
* `02\_Statistical\_Analysis/` — LSTM–GRU comparison, persistence benchmarking, statistical tests, bootstrap uncertainty, residual diagnostics, and high-water analysis.
* `03\_Figure\_Generation/` — scripts used to generate manuscript/supplementary figures.
* `requirements.txt` — Python dependencies.
* `CITATION.cff` — software citation metadata.
* `CODE\_AVAILABILITY.md` — manuscript code-availability wording.

## Data organization

Raw hydrological observations are **not included** in this repository. Place the three station files in a local `data/` directory, or provide another directory with `--data-dir`:

```text
data/
├── Hourly\_Water\_Level\_1984\_2022\_Tan\_Chau\_station.xlsx
├── Hourly\_Water\_Level\_1984\_2022\_Vam\_Nao\_station.xlsx
└── Hourly\_Water\_Level\_1984\_2022\_Can\_Tho\_station.xlsx
```

Model prediction files and analysis outputs are written under `outputs/` by default.

## Installation

Recommended: Python 3.13. Install the pinned dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt` pins each package to the latest stable PyPI release as of the study's
revision date (September 2026), per the authors' confirmation that the most recent available
version of each library was used at the time of analysis. Note that TensorFlow 2.21 does not
yet support Python 3.14 at the time of writing, so Python 3.13 is recommended over the newest
Python release.

The scripts no longer depend on Google Colab upload/download APIs. They use local paths and command-line arguments so that the workflow can be run from a cloned GitHub repository.

## Model configuration

The training scripts implement the study configuration:

* chronological 80/20 training/test split;
* MinMax scaling fitted on the training period only;
* direct multi-horizon point forecasting;
* forecast horizons: 1, 24, 48, 72, 96, 120, 144, and 168 h;
* horizon-specific historical lag windows: 94, 138, 148, 152, 220, 262, 288, and 302 h;
* two recurrent layers: 64 → 32 units;
* dropout = 0.20;
* Adam optimizer;
* learning rate = 1e-4;
* MSE loss;
* 50 epochs;
* batch size = 64;
* gradient clipping: `clipnorm=1.0`;
* `shuffle=False`;
* random seed = 42;
* no separate validation set and no early stopping.

The LSTM and GRU scripts use the same experimental configuration so that the recurrent-cell formulation is the principal architectural difference.

## Running the models

From the repository root:

```bash
python 01\_Model\_Training/LSTM/train\_lstm.py
python 01\_Model\_Training/GRU/train\_gru.py
```

To use another data location or output location:

```bash
python 01\_Model\_Training/LSTM/train\_lstm.py --data-dir /path/to/data --output-dir /path/to/lstm\_results
python 01\_Model\_Training/GRU/train\_gru.py --data-dir /path/to/data --output-dir /path/to/gru\_results
```

## Statistical analysis

After both model runs have produced their prediction files:

```bash
python 02\_Statistical\_Analysis/compare\_lstm\_gru.py
```

For additional analyses:

```bash
python 02\_Statistical\_Analysis/additional\_statistical\_analysis.py
```

Custom result locations can be supplied with `--lstm-dir`, `--gru-dir`, and `--output-dir`.

## Figure generation

Figure scripts now read forecast files from a directory rather than requiring interactive file uploads. For example:

```bash
python 03\_Figure\_Generation/Scatter/plot\_scatter\_observed\_vs\_predicted.py \\
  --input-dir outputs --model LSTM --station Can\_Tho

python 03\_Figure\_Generation/Violin/plot\_monthly\_violin.py \\
  --input-dir outputs --model LSTM --station Tan\_Chau

python 03\_Figure\_Generation/KDE/plot\_kde\_observed\_vs\_predicted.py \\
  --input-dir outputs --model LSTM --station Can\_Tho

python 03\_Figure\_Generation/NSE\_Heatmaps/plot\_monthly\_yearly\_nse\_heatmaps.py \\
  --input-dir outputs --model LSTM --station Tan\_Chau

python 03\_Figure\_Generation/Residual\_Diagnostics/plot\_residual\_diagnostics.py \\
  --input-dir outputs

python 03\_Figure\_Generation/Relative\_Performance/plot\_relative\_rmse\_gain\_loss\_gru\_vs\_lstm.py \\
  --input-dir outputs

python 03\_Figure\_Generation/Error\_Distribution/ERROR\_DISTRIBUTION.py \\
python 03\_Figure\_Generation/Peak\_Flood\_Event/plot\_peak\_flood\_event.py \\
  --input-dir outputs
```

Outputs are saved under `outputs/Figures/` by default.

## Reproducibility and scope

The repository contains the computational scripts required for the model training, statistical evaluation, and figure-generation workflow. Raw observations and potentially restricted study data are excluded. The repository therefore does not by itself guarantee numerical reproduction unless the same underlying input data and software environment are available.

## Citation

The `CITATION.cff` file provides machine-readable citation metadata; GitHub uses this file to expose a **Cite this repository** option.

For the journal submission, archive the GitHub release on Zenodo and add the resulting DOI to the manuscript's Code Availability statement.


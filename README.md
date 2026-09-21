# VMD-RWL-LSTM-GRU-Forecasting

Reproducible code accompanying the manuscript on multi-horizon hourly river water-level (RWL) forecasting using LSTM and GRU models at Vam Nao, Tan Chau, and Can Tho in the Vietnamese Mekong Delta.

## Repository contents

- `01_Model_Training/` — LSTM and GRU training scripts.
- `02_Statistical_Analysis/` — LSTM–GRU comparison, persistence benchmarking, statistical tests, bootstrap uncertainty, residual diagnostics, and high-water analysis.
- `03_Figure_Generation/` — scripts used to generate manuscript/supplementary figures.
- `requirements.txt` — Python dependencies.
- `CITATION.cff` — software citation metadata.
- `CODE_AVAILABILITY.md` — manuscript code-availability wording.

## Data organization

Raw hydrological observations are **not included** in this repository. Place the three station files in a local `data/` directory, or provide another directory with `--data-dir`:

```text
data/
├── Hourly_Water_Level_1984_2022_Tan_Chau_station.xlsx
├── Hourly_Water_Level_1984_2022_Vam_Nao_station.xlsx
└── Hourly_Water_Level_1984_2022_Can_Tho_station.xlsx
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

- chronological 80/20 training/test split;
- MinMax scaling fitted on the training period only;
- direct multi-horizon point forecasting;
- forecast horizons: 1, 24, 48, 72, 96, 120, 144, and 168 h;
- horizon-specific historical lag windows: 94, 138, 148, 152, 220, 262, 288, and 302 h;
- two recurrent layers: 64 → 32 units;
- dropout = 0.20;
- Adam optimizer;
- learning rate = 1e-4;
- MSE loss;
- 50 epochs;
- batch size = 64;
- gradient clipping: `clipnorm=1.0`;
- `shuffle=False`;
- random seed = 42;
- no separate validation set and no early stopping.

The LSTM and GRU scripts use the same experimental configuration so that the recurrent-cell formulation is the principal architectural difference.

## Running the models

From the repository root:

```bash
python 01_Model_Training/LSTM/train_lstm.py
python 01_Model_Training/GRU/train_gru.py
```

To use another data location or output location:

```bash
python 01_Model_Training/LSTM/train_lstm.py --data-dir /path/to/data --output-dir /path/to/lstm_results
python 01_Model_Training/GRU/train_gru.py --data-dir /path/to/data --output-dir /path/to/gru_results
```

## Statistical analysis

After both model runs have produced their prediction files:

```bash
python 02_Statistical_Analysis/compare_lstm_gru.py
```

For additional analyses:

```bash
python 02_Statistical_Analysis/additional_statistical_analysis.py
```

Custom result locations can be supplied with `--lstm-dir`, `--gru-dir`, and `--output-dir`.

## Figure generation

Figure scripts now read forecast files from a directory rather than requiring interactive file uploads. For example:

```bash
python 03_Figure_Generation/Scatter/plot_scatter_observed_vs_predicted.py \
  --input-dir outputs --model LSTM --station Can_Tho

python 03_Figure_Generation/Violin/plot_monthly_violin.py \
  --input-dir outputs --model LSTM --station Tan_Chau

python 03_Figure_Generation/KDE/plot_kde_observed_vs_predicted.py \
  --input-dir outputs --model LSTM --station Can_Tho

python 03_Figure_Generation/NSE_Heatmaps/plot_monthly_yearly_nse_heatmaps.py \
  --input-dir outputs --model LSTM --station Tan_Chau

python 03_Figure_Generation/Residual_Diagnostics/plot_residual_diagnostics.py \
  --input-dir outputs

python 03_Figure_Generation/Relative_Performance/plot_relative_rmse_gain_loss_gru_vs_lstm.py \
  --input-dir outputs

python 03_Figure_Generation/Error_Distribution/ERROR_DISTRIBUTION.py \
python 03_Figure_Generation/Peak_Flood_Event/plot_peak_flood_event.py \
  --input-dir outputs
```

Outputs are saved under `outputs/Figures/` by default.

## Reproducibility and scope

The repository contains the computational scripts required for the model training, statistical evaluation, and figure-generation workflow. Raw observations and potentially restricted study data are excluded. The repository therefore does not by itself guarantee numerical reproduction unless the same underlying input data and software environment are available.

## Citation

The `CITATION.cff` file provides machine-readable citation metadata; GitHub uses this file to expose a **Cite this repository** option.

For the journal submission, archive the GitHub release on Zenodo and add the resulting DOI to the manuscript's Code Availability statement.
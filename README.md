# Chronos Time Series Analysis

This repository contains scripts for analyzing time series energy data using a language model based method.

## Files Description

### requirements.txt 
This is an auto-generated file that lists all python package requirements for running the code in this project.

### chronos_wrapper.py
This file contains the code for using the Chronos language model for predicting TS data. 

### sarima_wrapper.py
Here you can find the code for generating forecast using the Seasonal - Auto Regressive Integrated Moving Average method. It is implemented using the `statsmodels.tsa.statespace.sarimax` package.

### lstm_wrapper.py
Code for predicting values using a LSTM based method. Data is normalized using a MinMaxScaler. A LSTM layer is followed by a linear layer. The LSTM layer processes the sequential input data (input_seq), maintaining an internal state (cell state and hidden state) that captures long-term dependencies in the data. During training, the model is trying to optimize MSE.

### gp_wrapper.py
Code for using a Gaussian Process based method for forecasting. Uses the `sklearn.gaussian_process` package.

### Mackey-Glass.py
This script generates a Mackey-Glass chaotic data series and saves it as a CSV file. The Mackey-Glass time series is a well-known chaotic dataset often used for testing time series forecasting models. You don't need to run it if  `mackey_glass_time_series.csv` already exists.

### mackey_glass_time_series.csv
This CSV file contains the generated Mackey-Glass chaotic time series data. It is used as input data for testing forecasting methods.

### kickoff_sliding_window.py
Modify and run this file to get cumulative errors for an algorithm across a dataset

Format
```
    python kickoff_sliding_window.py -a <algorithm> -d <data segment>
```

Example
```
    python kickoff_sliding_window.py -a chronos_tiny -d alpha
```

```
    python kickoff_sliding_window.py -a chronos_large -d alpha && python kickoff_sliding_window.py -a chronos_large -d beta && python kickoff_sliding_window.py -a chronos_large -d delta


```

### system_price_overview.ipynb
See overview of system prices on different time horizons

### system_price_forecasting.ipynb
Compare performance of SARIMA, Chronos, LSTM, and Gaussian Processes forecasting on System Energy Price Data

### half_hourly_prices_forecasting.ipynb
Compare performance of forecasting algorithms on half hourly energy prices of electricity. Provided by Octopus

### python tabulate_results.py 
Obtain tabulated results in latex code for performance measures across desired algorithms and data segment

```
 python tabulate_results.py --algorithm_names <desired algorithms> --data_segment <data segment of choice>
```


Examples:
```
 python tabulate_results.py --algorithm_names chronos_mini gp arima --data_segment delta

python tabulate_results.py --algorithm_names chronos_tiny arima gp chronos-tiny-336-48-8_000-alpha chronos-tiny-336-48-8_000-beta chronos-tiny-336-48-8_000-delta chronos-tiny-336-48-8_000-abd  --data_segments alpha 

```

### python graph_results.py
Example:

Graph basic three models 
```
python graph_results.py --algorithm_names chronos_tiny arima gp --data_segments alpha --ledger_key ledger_nmse
```

```
python graph_results.py --algorithm_names  chronos_tiny chronos_mini chronos_small chronos_base chronos_large  --data_segments alpha
```

Graph all FT models
```
python graph_results.py --algorithm_names chronos_mini arima gp chronos-tiny-336-48-8_000-alpha chronos-tiny-336-48-8_000-beta chronos-tiny-336-48-8_000-delta chronos-tiny-336-48-8_000-abd --ledger_key ledger_nmse --data_segments alpha 
```

### Data/

   Folder that contains all time series data for predicting and comparing performance of forecasting methods

    `system_price.csv`
    - 1637 time steps (4.48 years) of price per kwh, each representing a daily price
    - Training set     :   0  ->  730   (2 years, 730 samples)
    - Test set Alpha   :  731 ->  1096  (1 year, 365 samples)
    - Test set Beta    : 1097 ->  1462  (1 year, 365 samples)

    Agile_Octopus_London.csv
    - Original file name: Agile_Octopus_C_London-AGILE-22-07-22
    - 31,208 time steps/ 1.78 years of per kwh, each representing a half hour price
    - Training set      :        0 -> 10,402   (217 days, 10,416 samples)
    - Test set Gamma    :   10,403 -> 20,805   (217 days, 10,416 samples)
    - Test set Delta    :   20,806 -> 31,208   (217 days, 10,416 samples)


## Getting Ready

To install the necessary dependencies for this project, follow these steps:

1. **Ensure you have Python and pip installed**: You can check if you have them installed by running the following commands in your terminal:

    ```sh
    python --version
    pip --version
    ```

2. **Navigate to the project directory**: Open your terminal and navigate to the directory where the `requirements.txt` file is located. For example:

    ```sh
    cd path/to/this/repo/oracle
    ```

3. **Install the dependencies**: Run the following command to install all the required packages listed in the `requirements.txt` file:

    With Anaconda
    ```sh
    bash install_reqs.sh
    ```
    
    Without Anaconda
    ```sh
    pip install -r requirements.txt
    ```

After completing these steps, all the necessary packages specified in the `requirements.txt` file should be installed and ready to use.


## Usage

- Open and run the python notebooks listed above to see comparisons of forecasts of energy data 


## OMIE Day-Ahead Price Data
| Spain + Portugal | 2023-01-01 | 2026-03-26 | Day-ahead market only |

## PJM Day-Ahead LMP Data

Downloaded via `download_pjm_prices.py` using the PJM Data Miner 2 API. Data is stored in `data/day_ahead_pjm_{pnode_id}.csv`.

### Column reference

| Column | Description |
|---|---|
| `timestamp_utc` | Hour-beginning timestamp in UTC |
| `pnode_id` | Numeric pricing node ID (e.g. 51291 = AECO) |
| `pnode_name` | Human-readable node name |
| `total_lmp_da` | Total day-ahead LMP ($/MWh) — the all-in price at this node |
| `system_energy_price_da` | Energy component ($/MWh) — system-wide marginal cost of energy, uniform across the grid |
| `congestion_price_da` | Congestion component ($/MWh) — cost/benefit from transmission constraints; can be negative when the node is downstream of a bottleneck |
| `marginal_loss_price_da` | Loss component ($/MWh) — adjustment for transmission line losses relative to the reference bus |

The following identity always holds:

```
total_lmp_da = system_energy_price_da + congestion_price_da + marginal_loss_price_da
```

## Nord Pool Day-Ahead Price Data

Downloaded via `download_nordpool_prices.py` using the [ENTSO-E Transparency Platform](https://transparency.entsoe.eu) REST API (`entsoe-py`). Requires a free API key stored as `ENTSOE_KEY` in `.env`. Data is stored in `data/day_ahead_nordpool_{area}.csv`. Historical data available from **2015-01-05** onwards.

### Usage

```sh
# Default: Norway NO2, 2015 → today
python download_nordpool_prices.py

# Specific zone and date range
python download_nordpool_prices.py --area FI --start 2018-01-01
python download_nordpool_prices.py --area SE_3 --start 2018-01-01 --end 2025-01-01
```

### Column reference

| Column | Description |
|---|---|
| `timestamp_utc` | Hour-beginning timestamp in UTC |
| `price_eur_mwh` | Day-ahead price in EUR/MWh |

### Bidding zone reference

| Area code | Country | Zone description |
|---|---|---|
| `NO_1` | Norway | Oslo / Southeast Norway |
| `NO_2` | Norway | Kristiansand / Southwest Norway |
| `NO_3` | Norway | Trondheim / Central Norway |
| `NO_4` | Norway | Tromsø / Northern Norway |
| `NO_5` | Norway | Bergen / West Norway |
| `SE_1` | Sweden | Luleå / Northern Sweden |
| `SE_2` | Sweden | Sundsvall / North-Central Sweden |
| `SE_3` | Sweden | Stockholm / South-Central Sweden (largest zone) |
| `SE_4` | Sweden | Malmö / Southern Sweden |
| `DK_1` | Denmark | Jutland & Funen / West Denmark (connects to Continental Europe) |
| `DK_2` | Denmark | Zealand / East Denmark (connects to Nordic grid) |
| `FI` | Finland | Finland (single zone) |
| `EE` | Estonia | Estonia (single zone) |
| `LV` | Latvia | Latvia (single zone) |
| `LT` | Lithuania | Lithuania (single zone) |
| `DE_LU` | Germany/Luxembourg | Germany & Luxembourg (single price zone) |
| `FR` | France | France (single zone) |
| `NL` | Netherlands | Netherlands (single zone) |
| `BE` | Belgium | Belgium (single zone) |
| `AT` | Austria | Austria (single zone) |
| `GB` | Great Britain | Great Britain (N2EX day-ahead market) |

## GB Day-Ahead Price Data

Downloaded via `download_gb_prices.py` using the [Elexon Insights API](https://bmrs.elexon.co.uk) (Market Index Data / APXMIDP). No API key required. Data is stored in `data/day_ahead_gb_APXMIDP_{MMYY}_{MMYY}.csv`. Available from **2018-01-01** onwards.

> **Note:** GB left the EU internal energy market on 1 Jan 2021, so ENTSO-E data for `GB` only covers up to Dec 2020. Use this script for 2021 onwards (or for the full range).

### Usage

```sh
# Full history from 2018 to today
python download_gb_prices.py

# From Brexit onwards only
python download_gb_prices.py --start 2021-01-01

# Custom range
python download_gb_prices.py --start 2022-01-01 --end 2024-12-31
```

### Column reference

| Column | Description |
|---|---|
| `timestamp_utc` | Period-start timestamp in UTC |
| `settlementDate` | GB settlement date (local) |
| `settlementPeriod` | Half-hour slot 1–48 (1 = 00:00–00:30 local) |
| `price_gbp_mwh` | Day-ahead clearing price in £/MWh |
| `volume` | Volume traded in MWh |
| `dataProvider` | Always `APXMIDP` (EPEX SPOT GB) |

Resolution is **half-hourly (30 min)** — 48 periods per day. Currency is **GBP**, not EUR.

## City Weather Data

Downloaded via `download_city_weather.py` using [Meteostat](https://meteostat.net). No API key required. Data is stored in `data/weather_{city}.csv`. Covers one representative city per Nord Pool bidding zone, plus Madrid and Lisbon (OMIE).

### Usage

```sh
# All 23 cities (2015 → today)
python download_city_weather.py

# Single city
python download_city_weather.py --cities london

# Multiple cities
python download_city_weather.py --cities oslo stockholm helsinki london

# Custom date range
python download_city_weather.py --start 2018-01-01 --cities berlin paris amsterdam
```

### City reference

| City key | Station | Bidding zone |
|---|---|---|
| `oslo` | Oslo Blindern | NO1 |
| `kristiansand` | Kristiansand Kjevik | NO2 |
| `trondheim` | Trondheim Vaernes | NO3 |
| `tromso` | Tromsø Airport | NO4 |
| `bergen` | Bergen Florida | NO5 |
| `lulea` | Luleå Airport | SE1 |
| `sundsvall` | Sundsvall-Härnösand | SE2 |
| `stockholm` | Stockholm Arlanda | SE3 |
| `malmo` | Malmö Airport | SE4 |
| `aarhus` | Aarhus (Tirstrup) | DK1 |
| `copenhagen` | Copenhagen Kastrup | DK2 |
| `helsinki` | Helsinki Vantaa | FI |
| `tallinn` | Tallinn Airport | EE |
| `riga` | Riga Airport | LV |
| `vilnius` | Vilnius Airport | LT |
| `berlin` | Berlin-Tegel | DE_LU |
| `paris` | Paris Charles de Gaulle | FR |
| `amsterdam` | Amsterdam Schiphol | NL |
| `brussels` | Brussels Zaventem | BE |
| `vienna` | Vienna Schwechat | AT |
| `london` | London Heathrow | GB |
| `madrid` | Madrid-Barajas | OMIE (Spain) |
| `lisbon` | Lisbon Humberto Delgado | OMIE (Portugal) |

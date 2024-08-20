# Alpha
python3 graph_results.py --algorithm_names chronos_tiny arima gp --data_segments alpha --ledger_key ledger_nmse --run_name nmse_alpha_basic
python3 graph_results.py --algorithm_names chronos_tiny chronos_mini chronos_small chronos_base chronos_large --data_segments alpha --ledger_key ledger_nmse --run_name nmse_alpha_chronos

# Beta
python3 graph_results.py --algorithm_names chronos_tiny arima gp --data_segments beta --ledger_key ledger_nmse --run_name nmse_beta_basic
python3 graph_results.py --algorithm_names chronos_tiny chronos_mini chronos_small chronos_base chronos_large --data_segments beta --ledger_key ledger_nmse --run_name nmse_beta_chronos

# Delta
python3 graph_results.py --algorithm_names chronos_tiny arima gp --data_segments delta --ledger_key ledger_nmse --run_name nmse_delta_basic
python3 graph_results.py --algorithm_names chronos_tiny chronos_mini chronos_small chronos_base chronos_large --data_segments delta --ledger_key ledger_nmse --run_name nmse_delta_chronos

# FT - Alpha
python3 graph_results.py --algorithm_names chronos_tiny arima gp chronos-tiny-336-48-8_000-alpha chronos-tiny-336-48-8_000-beta chronos-tiny-336-48-8_000-delta chronos-tiny-336-48-8_000-abd  --data_segments alpha --ledger_key ledger_nmse --run_name nmse_alpha_ft

 # FT - Beta
python3 graph_results.py --algorithm_names chronos_tiny arima gp chronos-tiny-336-48-8_000-alpha chronos-tiny-336-48-8_000-beta chronos-tiny-336-48-8_000-delta chronos-tiny-336-48-8_000-abd  --data_segments beta --ledger_key ledger_nmse --run_name nmse_beta_ft

  # FT - Delta
python3 graph_results.py --algorithm_names chronos_tiny arima gp chronos-tiny-336-48-8_000-alpha chronos-tiny-336-48-8_000-beta chronos-tiny-336-48-8_000-delta chronos-tiny-336-48-8_000-abd  --data_segments delta --ledger_key ledger_nmse --run_name nmse_delta_ft
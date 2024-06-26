import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def mackey_glass(t_max, tau, beta=0.2, gamma=0.1, n=10, delta_t=0.1):
    t_steps = int(t_max / delta_t)
    tau_steps = int(tau / delta_t)
    
    x = np.zeros(t_steps + 1)
    x[0] = 1.2  # Initial condition
    
    for t in range(t_steps):
        if t < tau_steps:
            x[t + 1] = x[t] + delta_t * (beta * x[t] / (1 + x[t]**n) - gamma * x[t])
        else:
            x[t + 1] = x[t] + delta_t * (beta * x[t - tau_steps] / (1 + x[t - tau_steps]**n) - gamma * x[t])
    
    return x

# Parameters
t_max = 1000
tau = 17

# Generate Mackey-Glass time series
x = mackey_glass(t_max, tau)

# Create a DataFrame
time_series_df = pd.DataFrame({
    'Time': np.arange(0, t_max + 0.1, 0.1),
    'Value': x
})

# Save to CSV
time_series_df.to_csv('mackey_glass_time_series.csv', index=False)

# Plot the time series
plt.figure(figsize=(10, 5))
plt.plot(time_series_df['Time'], time_series_df['Value'])
plt.title('Mackey-Glass Time Series')
plt.xlabel('Time')
plt.ylabel('x(t)')
plt.show()
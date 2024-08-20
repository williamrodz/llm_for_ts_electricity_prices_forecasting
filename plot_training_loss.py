import re
import matplotlib.pyplot as plt

# Function to read log text from a file
def read_log_file(file_path):
    with open(file_path, 'r') as file:
        log_text = file.read()
    return log_text

# Path to the log file
def plot_training_loss(file_name):
        
    log_file_path = f'training_logs/{file_name}.txt'

    # Read the log text from the file
    log_text = read_log_file(log_file_path)

    # Regular expression pattern to extract the relevant data
    pattern = r"\{'loss': ([\d\.]+), .* 'epoch': ([\d\.]+)\}"

    # Find all matches in the log text
    matches = re.findall(pattern, log_text)

    # Extract loss and epoch values
    loss_values = [float(match[0]) for match in matches]
    epoch_values = [float(match[1]) for match in matches]

    # Plotting the loss over time (epoch)
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_values, loss_values, marker='o', linestyle='-', color='purple')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # plt.title('Loss over Time')
    plt.grid(True)
    plt.savefig(f'results/plots/{file_name}.png', dpi=300)
    plt.show()

for file_name in ['chronos-tiny-336-48-8_000-alpha', 'chronos-tiny-336-48-8_000-beta', 'chronos-tiny-336-48-8_000-delta','chronos-tiny-336-48-8_000-abd']:
    plot_training_loss(file_name)

import psutil
import time
import statistics
import csv
import datetime

def log_metrics(interval, duration, filename):
    cpu_usages = []
    memory_usages = []

    start_time = time.time()
    end_time = start_time + duration

    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Time", "CPU Usage (%)", "Memory Usage (MB)"])

        print(f"{'Time':<10}{'CPU Usage (%)':<15}{'Memory Usage (MB)':<20}")
        print("-" * 45)

        while time.time() < end_time:
            cpu_usage = psutil.cpu_percent(interval=interval)
            memory_info = psutil.virtual_memory()
            memory_usage = memory_info.used / (1024 * 1024)  # Convert bytes to MB

            cpu_usages.append(cpu_usage)
            memory_usages.append(memory_usage)

            current_time = time.strftime('%H:%M:%S')
            writer.writerow([current_time, cpu_usage, memory_usage])

            print(f"{current_time:<10}{cpu_usage:<15}{memory_usage:<20.2f}")

    max_cpu_usage = max(cpu_usages)
    avg_cpu_usage = statistics.mean(cpu_usages)
    max_memory_usage = max(memory_usages)

    print("\nMetrics Summary:")
    print(f"Max CPU Usage: {max_cpu_usage:.2f}%")
    print(f"Avg CPU Usage: {avg_cpu_usage:.2f}%")
    print(f"Max Memory Usage: {max_memory_usage:.2f} MB")

if __name__ == "__main__":
    interval = 1  # Interval between measurements in seconds
    duration = 30  # Duration of logging in seconds
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"metrics_log_{timestamp}.csv"  # Output file name with timestamp

    log_metrics(interval, duration, filename)

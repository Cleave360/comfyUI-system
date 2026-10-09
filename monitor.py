import psutil
import time
import os

def print_memory_dashboard():
    os.system('cls' if os.name == 'nt' else 'clear')

    mem = psutil.virtual_memory()
    total_gb = mem.total / (1024**3)
    used_gb = mem.used / (1024**3)
    available_gb = mem.available / (1024**3)

    pressure = (used_gb / total_gb) * 100

    print("--- 🖥️  M3 ULTRA 512GB UNIFIED MEMORY DASHBOARD ---")
    print(f"Total Capacity: {total_gb:.2f} GB")
    print(f"Current Usage:  {used_gb:.2f} GB [{'█' * int(pressure//5)}{'-' * (20 - int(pressure//5))}] {pressure:.1f}%")
    print(f"Available:      {available_gb:.2f} GB")
    print("--------------------------------------------------")

    if available_gb < 50:
        print("⚠️  STATUS: High Pressure. Gemini should pause model swaps.")
    else:
        print("✅ STATUS: Optimal. Room for multiple FP16 parallel loads.")

if __name__ == "__main__":
    try:
        while True:
            print_memory_dashboard()
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

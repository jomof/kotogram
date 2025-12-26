#!/usr/bin/env python3
"""
Standalone watchdog script for profiling training processes.
Run with sudo: sudo python3 scripts/watchdog.py
"""
import os
import sys
import time
import subprocess
import glob
import re

def find_target_pids():
    """Find all python processes related to train_style."""
    pids = set()
    try:
        # Check all processes
        # pgrep -f "train_style" is simple but might catch itself or editors
        # We look for python processes running train_style or its children
        out = subprocess.check_output(["pgrep", "-f", "train_style"], text=True)
        for pid_str in out.strip().split():
            if not pid_str: continue
            pid = int(pid_str)
            if pid == os.getpid(): continue
            pids.add(pid)
    except subprocess.CalledProcessError:
        pass
    
    return sorted(list(pids))

def run_watchdog():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    profile_dir = os.path.join(project_root, ".profile")
    os.makedirs(profile_dir, exist_ok=True)
    output_file = os.path.join(profile_dir, "process-threads.txt")
    
    print(f"Watchdog started. Monitoring for 'train_style' processes.")
    print(f"Output: {output_file}")
    print("Press Ctrl+C to stop.")

    while True:
        try:
            pids = find_target_pids()
            if not pids:
                sys.stdout.write(".")
                sys.stdout.flush()
                time.sleep(5)
                continue

            output = []
            for pid in pids:
                output.append(f"\n===== PID {pid} =====")
                try:
                    # py-spy dump
                    # check if pid still exists
                    try:
                        os.kill(pid, 0)
                    except OSError:
                        continue

                    dump = subprocess.check_output(
                        ["py-spy", "dump", "-p", str(pid)], 
                        text=True, 
                        stderr=subprocess.STDOUT
                    )
                    output.append(dump)
                except FileNotFoundError:
                     output.append("[py-spy not found]")
                except subprocess.CalledProcessError as e:
                    output.append(f"[py-spy failed: {e.output.strip()}]")
                except Exception as e:
                    output.append(f"[error: {e}]")

            if output:
                 timestamp = time.ctime()
                 entry = f"Watchdog Dump at {timestamp}\n" + "\n".join(output)
                 # Atomic write
                 temp_path = output_file + ".tmp"
                 with open(temp_path, "w") as f:
                     f.write(entry)
                 os.replace(temp_path, output_file)
                 sys.stdout.write(f"\rUpdated dump for {len(pids)} processes at {timestamp}   ")
                 sys.stdout.flush()
            
            time.sleep(5)

        except KeyboardInterrupt:
            print("\nWatchdog stopped.")
            break
        except Exception as e:
            print(f"\nError: {e}")
            time.sleep(5)


if __name__ == "__main__":
    if os.geteuid() != 0:
        print("Warning: Not running as root. py-spy might fail to attach.")
    run_watchdog()

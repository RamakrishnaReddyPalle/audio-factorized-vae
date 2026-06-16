# src/trainers/memory_monitor.py

import os
import time
import psutil


class MemoryMonitor:

    def __init__(self):

        self.process = psutil.Process(
            os.getpid()
        )

        self.start_time = None

    def start_epoch(self):

        self.start_time = time.time()

    def get_ram_gb(self):

        return (

            self.process
            .memory_info()
            .rss

            /

            1024**3
        )

    def get_cpu_percent(self):

        return psutil.cpu_percent()

    def end_epoch(self):

        elapsed = (

            time.time()
            -
            self.start_time
        )

        return {

            "epoch_time":
                elapsed,

            "ram_gb":
                self.get_ram_gb(),

            "cpu_percent":
                self.get_cpu_percent()
        }
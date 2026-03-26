import json
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from time import time

import numpy as np
import torch


class Benchmarker:
    def __init__(self):
        self.execution_times = defaultdict(list)
        self.benchmarks = {}

    @contextmanager
    def time(self, tag: str, num_calls: int = 1):
        try:
            start_time = time()
            yield
        finally:
            end_time = time()
            for _ in range(num_calls):
                self.execution_times[tag].append((end_time - start_time) / num_calls)

    def store(self, tag, x):
        if tag not in self.benchmarks.keys():
            self.benchmarks[tag] = []
        self.benchmarks[tag].append(x)

    def dump(self, path: Path) -> None:
        path.parent.mkdir(exist_ok=True, parents=True)
        benchmark_dict = dict(self.execution_times)
        mean_encoder_time = float(np.mean(benchmark_dict["encoder"])) if "encoder" in benchmark_dict and len(benchmark_dict["encoder"]) > 0 else float("nan")
        mean_decoder_time = float(np.mean(benchmark_dict["decoder"])) if "decoder" in benchmark_dict and len(benchmark_dict["decoder"]) > 0 else float("nan")
        fps = float(1.0 / mean_decoder_time) if np.isfinite(mean_decoder_time) and mean_decoder_time > 0 else float("nan")
        benchmark_dict["mean_encoder_time"] = mean_encoder_time
        benchmark_dict["mean_decoder_time"] = mean_decoder_time
        benchmark_dict["fps"] = fps
        with path.open("w") as f:
            json.dump(benchmark_dict, f)
    
    def dump_stats(self, path: Path) -> None:
        path.parent.mkdir(exist_ok=True, parents=True)
        new_benchmarks = self.benchmarks.copy()
        for tag in self.benchmarks.keys():
            new_benchmarks[tag+'_avg'] = np.nanmean(np.asarray(self.benchmarks[tag], dtype=float))
        with path.open("w") as f:
            json.dump(new_benchmarks, f)
        self.benchmarks = new_benchmarks

    def dump_memory(self, path: Path) -> None:
        path.parent.mkdir(exist_ok=True, parents=True)
        with path.open("w") as f:
            json.dump(torch.cuda.memory_stats()["allocated_bytes.all.peak"], f)

    def summarize(self) -> None:
        for tag, times in self.execution_times.items():
            print(f"{tag}: {len(times)} calls, avg. {np.mean(times)} seconds per call")

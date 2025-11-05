# gpu_launcher.py
import os
import sys
import time
import pynvml
import argparse

def get_all_gpu_memory():
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    memory_list = []
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used_mb = mem_info.used // (1024 * 1024)
        memory_list.append(used_mb)
    pynvml.nvmlShutdown()
    return memory_list

def wait_for_free_gpus(required_gpu_count=2, memory_threshold=10000, interval=2):
    print(f"Waiting for {required_gpu_count} free GPU(s) (<= {memory_threshold} MiB used)...")
    count = 0
    while True:
        try:
            memories = get_all_gpu_memory()
        except Exception as e:
            print(f"\nError reading GPU info: {e}")
            time.sleep(interval)
            continue

        free_gpus = [i for i, mem in enumerate(memories) if mem <= memory_threshold]
        if len(free_gpus) >= required_gpu_count:
            selected_gpus = free_gpus[:required_gpu_count]
            print(f"\nFound enough free GPUs: {selected_gpus}")
            return selected_gpus

        print(f"count: {count}", end='\r')
        count += 1
        status_str = " | ".join([f"GPU{i}:{mem}MiB" for i, mem in enumerate(memories)])
        sys.stdout.write(f'\rWaiting... {status_str} | Free: {free_gpus}')
        sys.stdout.flush()
        time.sleep(interval)

def set_cuda_visible_devices(gpu_ids):
    gpu_str = ",".join(map(str, gpu_ids))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
    print(f"[INFO] Set CUDA_VISIBLE_DEVICES={gpu_str}")

def main(cmd='bash ./train.sh', required_gpu_count=4, memory_threshold=25000, specified_gpus=None):
    if specified_gpus is not None:
        print(f"[INFO] Using specified GPUs: {specified_gpus}")
        set_cuda_visible_devices(specified_gpus)
    else:
        free_gpus = wait_for_free_gpus(
            required_gpu_count=required_gpu_count,
            memory_threshold=memory_threshold,
            interval=2
        )
        set_cuda_visible_devices(free_gpus)
    
    # 直接执行命令（和你原来一样）
    exit_code = os.system(cmd)
    if exit_code != 0:
        sys.exit(exit_code)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', nargs='*', type=int, help="Specify GPU IDs, e.g., --gpus 0 1 2 3")
    parser.add_argument('--count', type=int, default=8)
    parser.add_argument('--threshold', type=int, default=13000)
    args = parser.parse_args()

    # 执行两个任务（顺序）
    main(cmd='bash ./train_GRNet.sh', required_gpu_count=args.count, memory_threshold=args.threshold, specified_gpus=args.gpus)
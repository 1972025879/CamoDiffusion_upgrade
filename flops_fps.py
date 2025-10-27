import torch
import torch.nn as nn
import time
from ptflops import get_model_complexity_info
from model.net import net

def input_constructor(input_res):
    batch_size = 1
    C, H, W = 3, input_res[0], input_res[1]
    
    # 注意：你这里 x 是 1 通道？根据你的模型决定
    x = torch.randn(batch_size, 1, H, W).cuda()          # ← 确认是否应为 1 通道
    timesteps = torch.randint(0, 1000, (batch_size,)).cuda()
    cond_img = torch.randn(batch_size, C, H, W).cuda()   # 3 通道条件图

    return {
        "x": x,
        "timesteps": timesteps,
        "cond_img": cond_img
    }

def get_dummy_input(input_res=(352, 352)):
    """用于 FPS 测试的输入（与 input_constructor 一致）"""
    batch_size = 1
    C, H, W = 3, input_res[0], input_res[1]
    x = torch.randn(batch_size, 1, H, W).cuda()
    timesteps = torch.randint(0, 1000, (batch_size,)).cuda()
    cond_img = torch.randn(batch_size, C, H, W).cuda()
    return x, timesteps, cond_img

if __name__ == "__main__":
    input_resolution = (352, 352)
    
    # === 1. 测 FLOPs 和 Params ===
    model = net(class_num=2, mask_chans=1).cuda()
    model.eval()

    macs, params = get_model_complexity_info(
        model,
        input_res=input_resolution,
        input_constructor=input_constructor,
        as_strings=True,
        print_per_layer_stat=False,
        verbose=True,
        backend='aten'
    )
    print(f"{'计算复杂度 (MACs):':<30} {macs}")
    print(f"{'参数量 (Params):':<30} {params}")

    # === 2. 测 FPS ===
    x, timesteps, cond_img = get_dummy_input(input_resolution)
    
    # 预热（warm-up）
    with torch.no_grad():
        for _ in range(10):
            _ = model(x, timesteps, cond_img)
    
    torch.cuda.synchronize()
    start_time = time.time()
    num_runs = 100  # 可调整，建议 50~200
    
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(x, timesteps, cond_img)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    total_time = end_time - start_time
    fps = num_runs / total_time

    print(f"{'推理速度 (FPS):':<30} {fps:.2f} FPS")
    print(f"{'单次推理耗时:':<30} {1000.0 / fps:.2f} ms")
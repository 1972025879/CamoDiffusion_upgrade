import torch
import torch.nn as nn
import time
from ptflops import get_model_complexity_info
from model.net import net
class ModelAnalyzer:
    """
    用于分析 PyTorch 模型的性能指标，包括 FLOPs、参数量和 FPS。
    """
    def __init__(self, model: nn.Module, input_res: tuple = (352, 352), batch_size: int = 32):
        """
        初始化分析器。

        Args:
            model (nn.Module): 要分析的 PyTorch 模型。
            input_res (tuple): 输入图像的分辨率 (H, W)。
            batch_size (int): 分析时使用的批次大小，默认为 1。
        """
        self.model = model
        self.input_res = input_res
        self.batch_size = batch_size
        self.device = next(model.parameters()).device # 自动获取模型所在的设备

    def _input_constructor(self,input_res): #定义输入net的数据格式
        """
        输入构造函数，用于 ptflops 库。
        """
        C, H, W = 3, self.input_res[0], self.input_res[1] # 假设 cond_img 是 3 通道
        x = torch.randn(self.batch_size, 1, H, W, device=self.device)
        timesteps = torch.randint(0, 1000, (self.batch_size,), device=self.device)
        cond_img = torch.randn(self.batch_size, C, H, W, device=self.device)

        return {
            "x": x,
            "timesteps": timesteps,
            "cond_img": cond_img
        }

    def _get_dummy_input(self):
        """
        生成用于 FPS 测试的输入张量。
        """
        C, H, W = 3, self.input_res[0], self.input_res[1]
        x = torch.randn(self.batch_size, 1, H, W, device=self.device)
        timesteps = torch.randint(0, 1000, (self.batch_size,), device=self.device)
        cond_img = torch.randn(self.batch_size, C, H, W, device=self.device)
        return x, timesteps, cond_img
    
    def one_image_try(self):
        """
        用于模型测试：向模型输入一张图片
        """
        self.model.eval()
        x, timesteps, cond_img = self._get_dummy_input()
        self.model(x, timesteps, cond_img)
        
    def analyze_flops_params(self) -> tuple:
        """
        分析模型的 FLOPs 和参数量。

        Returns:
            tuple: (flops_str, params_str) 分别是 FLOPs 和参数量的字符串表示。
        """
        # 确保模型在评估模式
        self.model.eval()
        
        flops, params = get_model_complexity_info(
            self.model,
            input_res=self.input_res,
            input_constructor=self._input_constructor,
            as_strings=True,
            print_per_layer_stat=False,
            verbose=False, # 设为 False 以减少 ptflops 的输出
            backend='aten'
        )
        return flops, params

    def measure_fps(self, num_runs: int = 100, warmup_runs: int = 10) -> float:
        """
        测量模型的推理速度 (FPS)。

        Args:
            num_runs (int): 用于计算 FPS 的推理运行次数。
            warmup_runs (int): 预热运行次数。

        Returns:
            float: 模型的平均推理速度 (FPS)。
        """
        self.model.eval()
        x, timesteps, cond_img = self._get_dummy_input()

        # 预热
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = self.model(x, timesteps, cond_img)

        torch.cuda.synchronize(device=self.device)
        start_time = time.time()

        with torch.no_grad():
            for _ in range(num_runs):
                _ = self.model(x, timesteps, cond_img)

        torch.cuda.synchronize(device=self.device)
        end_time = time.time()

        total_time = end_time - start_time
        fps = num_runs / total_time
        return fps

    def run_complete_analysis(self, num_fps_runs: int = 100, fps_warmup_runs: int = 10):
        """
        运行完整的性能分析，包括 FLOPs, Params 和 FPS。

        Args:
            num_fps_runs (int): 用于计算 FPS 的推理运行次数。
            fps_warmup_runs (int): FPS 测试的预热运行次数。
        """
        print(f"--- 模型性能分析 ---")
        print(f"输入分辨率: {self.input_res}")
        print(f"批次大小: {self.batch_size}")
        print(f"设备: {self.device}")
        print("-" * 40)

        # 1. 分析 FLOPs 和 Params
        flops, params = self.analyze_flops_params()
        print(f"{'计算复杂度 (MACs):':<30} {flops}")
        print(f"{'参数量 (Params):':<30} {params}")

        # 2. 测量 FPS
        fps = self.measure_fps(num_runs=num_fps_runs, warmup_runs=fps_warmup_runs)
        print(f"{'推理速度 (FPS):':<30} {fps:.2f} FPS")
        print(f"{'单次推理耗时:':<30} {1000.0 / fps:.2f} ms")
        print("-" * 40)


if __name__ == "__main__":
    # 1. 创建你的模型实例
    model = net(class_num=2, mask_chans=1).cuda() # 假设 net 是从 model.net 导入的

    # 2. 创建分析器实例
    analyzer = ModelAnalyzer(model, input_res=(352, 352), batch_size=1)
    
    # 3. 一张图片试运行
    analyzer.one_image_try()
    
    # 4. 运行完整分析
    analyzer.run_complete_analysis(num_fps_runs=100, fps_warmup_runs=10)

    # 或者，你也可以分别调用：
    # flops, params = analyzer.analyze_flops_params()
    # fps = analyzer.measure_fps(num_runs=100, warmup_runs=10)
    # print(f"FLOPs: {flops}, Params: {params}, FPS: {fps:.2f}")
import sys

import torch
from utils.train_utils import set_random_seed

from utils import init_env
import os
import argparse
from pathlib import Path

from utils.collate_utils import collate
from utils.import_utils import instantiate_from_config, recurse_instantiate_from_config, get_obj_from_str
from utils.init_utils import add_args
from torch.utils.data import DataLoader
from utils.trainer import Trainer

set_random_seed(7)


def get_loader(cfg):
    # 修改 1: 获取 ORSSD 数据集加载器
    orssd_test_dataset = instantiate_from_config(cfg.test_dataset.ORSSD)
    
    # 修改 2: 创建 ORSSD 数据集加载器
    orssd_test_loader = DataLoader(
        orssd_test_dataset,
        batch_size=cfg.batch_size,
        collate_fn=collate
    )
    # 修改 3: 只返回 ORSSD 加载器
    return orssd_test_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--results_folder', type=str, default='./results')
    parser.add_argument('--num_epoch', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gradient_accumulate_every', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--num_sample_steps', type=int, default=None)
    # 修改 4: 默认目标数据集改为 ORSSD
    parser.add_argument('--target_dataset', nargs='+', type=str, default=['ORSSD'])
    parser.add_argument('--time_ensemble', action='store_true')
    parser.add_argument('--batch_ensemble', action='store_true')
    parser.add_argument('--custom_header', type=str, default="CamoDiffusion")
    
    
    cfg = add_args(parser)
    assert not (cfg.time_ensemble and cfg.batch_ensemble), 'Cannot use both time_ensemble and batch_ensemble'
    """
        Hack config here.
    """
    if cfg.num_sample_steps is not None:
        cfg.diffusion_model.params.num_sample_steps = cfg.num_sample_steps

    # 修改 5: 接收 ORSSD 加载器
    orssd_test_loader = get_loader(cfg)

    cond_uvit = instantiate_from_config(cfg.cond_uvit,
                                        conditioning_klass=get_obj_from_str(cfg.cond_uvit.params.conditioning_klass))
    model = recurse_instantiate_from_config(cfg.model,
                                            unet=cond_uvit)

    diffusion_model = instantiate_from_config(cfg.diffusion_model,
                                              model=model)

    optimizer = instantiate_from_config(cfg.optimizer, params=model.parameters())

    trainer = Trainer(
        diffusion_model,
        train_loader=None, test_loader=None,
        train_val_forward_fn=get_obj_from_str(cfg.train_val_forward_fn),
        gradient_accumulate_every=cfg.gradient_accumulate_every,
        results_folder=cfg.results_folder,
        optimizer=optimizer,
        train_num_epoch=cfg.num_epoch,
        amp=cfg.fp16,
        log_with=None,
        cfg=cfg,
    )

    trainer.load(pretrained_path=cfg.checkpoint)
    # 修改 6: 准备 ORSSD 加载器
    orssd_test_loader = trainer.accelerator.prepare(orssd_test_loader)

    # 修改 7: 更新数据集映射字典，只包含 ORSSD
    dataset_map = {
        'ORSSD': orssd_test_loader,
    }
    # 修改 8: 检查目标数据集是否在映射中
    assert all([d_name in dataset_map.keys() for d_name in cfg.target_dataset]), \
        f'Invalid dataset name. Available dataset: {dataset_map.keys()}' \
        f'Your input: {cfg.target_dataset}'
    target_dataset = [(dataset_map[dataset_name], dataset_name) for dataset_name in cfg.target_dataset]

    for dataset, dataset_name in target_dataset:
        trainer.model.eval()
        # 修改 9: 获取 ORSSD 的 image_root 路径
        mask_path = Path(cfg.test_dataset.ORSSD.params.image_root).parent.parent
        save_to = Path(cfg.results_folder) / dataset_name
        os.makedirs(save_to, exist_ok=True)
        if cfg.batch_ensemble:
            mae, _ = trainer.val_batch_ensemble(model=trainer.model,
                                                test_data_loader=dataset,
                                                accelerator=trainer.accelerator,
                                                thresholding=False,
                                                save_to=save_to)
        elif cfg.time_ensemble:
            mae, _ = trainer.val_time_ensemble(model=trainer.model,
                                               test_data_loader=dataset,
                                               accelerator=trainer.accelerator,
                                               thresholding=False,
                                               save_to=save_to)
        else:
            mae, _ = trainer.val(model=trainer.model,
                                 test_data_loader=dataset,
                                 accelerator=trainer.accelerator,
                                 thresholding=False,
                                 save_to=save_to)
        trainer.accelerator.wait_for_everyone()
        trainer.accelerator.print(f'{dataset_name} mae: {mae}')

        if trainer.accelerator.is_main_process:
            from utils.eval import eval2

            eval_score = eval2(
                mask_path=mask_path,
                pred_path=cfg.results_folder,
                dataset_name=dataset_name)
            # === 新增：将 eval_score 保存到 txt 文件 ===
            results_file = Path(cfg.results_folder) / "evaluation_results.txt"
            custom_header = cfg.custom_header  # 从命令行参数获取
            with open(results_file, "a") as f:
                f.write(custom_header + "\n")
                f.write(f"{dataset_name}:\n")
                # 假设 eval_score 是一个字典，例如 {'Smeasure': 0.85, 'wFmeasure': 0.76, ...}
                if isinstance(eval_score, dict):
                    for key, value in eval_score.items():
                        f.write(f"  {key}: {value:.6f}\n")
                else:
                    f.write(f"  {str(eval_score)}\n")
                f.write("\n")  # 空行分隔不同数据集
        trainer.accelerator.wait_for_everyone()

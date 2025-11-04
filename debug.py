# debug_eorssd.py
import os
import numpy as np
from PIL import Image
from tqdm import tqdm

def check_eorssd_dataset(image_root, gt_root, dataset_name="EORSSD"):
    print(f"🔍 开始调试 {dataset_name} 数据集...")

    # 获取所有图像和 GT 文件
    image_files = [f for f in os.listdir(image_root) if f.endswith(('.jpg', '.png'))]
    gt_files = [f for f in os.listdir(gt_root) if f.endswith(('.jpg', '.png', '.tif'))]

    # 检查文件名是否匹配
    image_names = set([os.path.splitext(f)[0] for f in image_files])
    gt_names = set([os.path.splitext(f)[0] for f in gt_files])

    mismatched = image_names - gt_names
    if mismatched:
        print(f"⚠️ 图像有但 GT 缺失: {len(mismatched)} 个")
        for name in sorted(mismatched)[:5]:
            print(f"   {name}")

    # 检查 GT 是否为全黑图
    black_count = 0
    size_mismatch_count = 0
    total_samples = len(image_files)

    for img_file in tqdm(image_files, desc="检查样本"):
        img_name = os.path.splitext(img_file)[0]
        gt_file = None

        # 尝试匹配 GT 文件（支持 .jpg/.png/.tif）
        for ext in ['.jpg', '.png', '.tif']:
            candidate = img_name + ext
            if candidate in gt_files:
                gt_file = candidate
                break

        if not gt_file:
            continue  # 无对应 GT，跳过

        # 加载 GT
        gt_path = os.path.join(gt_root, gt_file)
        try:
            gt = Image.open(gt_path).convert('L')
            gt_np = np.array(gt, dtype=np.float32)

            # 检查是否为全黑图
            if np.sum(gt_np) == 0:
                black_count += 1
                print(f"⚠️ 全黑图发现: {gt_file} | shape={gt_np.shape} | sum={np.sum(gt_np)}")

            # 检查尺寸（可选）
            if gt_np.shape[0] != 352 or gt_np.shape[1] != 352:
                size_mismatch_count += 1
                print(f"⚠️ 尺寸不匹配: {gt_file} | shape={gt_np.shape}")

        except Exception as e:
            print(f"❌ 加载失败: {gt_file} | 错误: {str(e)}")

    print("\n📊 调试报告:")
    print(f"总样本数: {total_samples}")
    print(f"全黑图数量: {black_count}")
    print(f"尺寸不匹配数量: {size_mismatch_count}")
    print(f"建议: 删除或修复上述样本后重新训练")

if __name__ == "__main__":
    train_image_root = 'media/dataset/COD10K/TrainDataset/Image/'
    train_gt_root = 'media/dataset/COD10K/TrainDataset/GT/'

    test_image_root = 'media/dataset/COD10K/TestDataset/NC4K/Image/'
    test_gt_root = 'media/dataset/COD10K/TestDataset/NC4K/GT/'

    print("✅ 训练集调试:")
    check_eorssd_dataset(train_image_root, train_gt_root, "Train")

    print("\n✅ 测试集调试:")
    check_eorssd_dataset(test_image_root, test_gt_root, "Test")
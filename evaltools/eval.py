import os
import sys
import cv2
import time
from tqdm import tqdm
import metrics
import json
import argparse
import numpy as np


def Borders_Capture(gt, pred, dksize=15):
    gray = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img = gt.copy()
    img[:] = 0
    cv2.drawContours(img, contours, -1, (255, 255, 255), 3)
    kernel = np.ones((dksize, dksize), np.uint8)
    img_dilate = cv2.dilate(img, kernel)

    res = cv2.bitwise_and(img_dilate, gt)
    b, g, r = cv2.split(res)
    alpha = np.rollaxis(img_dilate, 2, 0)[0]
    merge = cv2.merge((b, g, r, alpha))

    resp = cv2.bitwise_and(img_dilate, pred)
    b, g, r = cv2.split(resp)
    alpha = np.rollaxis(img_dilate, 2, 0)[0]
    mergep = cv2.merge((b, g, r, alpha))

    merge = cv2.cvtColor(merge, cv2.COLOR_RGB2GRAY)
    mergep = cv2.cvtColor(mergep, cv2.COLOR_RGB2GRAY)
    return merge, mergep, np.sum(img_dilate) / 255


def eval(args, dataset):
    FM = metrics.Fmeasure_and_FNR()
    WFM = metrics.WeightedFmeasure()
    SM = metrics.Smeasure()
    EM = metrics.Emeasure()
    MAE = metrics.MAE()
    BR_MAE = metrics.MAE()
    BR_wF = metrics.WeightedFmeasure()

    model = args.model
    gt_root = os.path.join(args.GT_root, dataset, 'GT')
    pred_root = os.path.join(args.pred_root, dataset)

    if not os.path.isdir(pred_root):
        print(f'[WARN] pred_root not exist: {pred_root}')
        return

    gt_name_list = sorted(os.listdir(pred_root))
    print(f'Processing dataset={dataset}, #preds={len(gt_name_list)}')

    for gt_name in tqdm(gt_name_list, total=len(gt_name_list), miniters=50):
        try:
            gt_path = os.path.join(gt_root, gt_name)
            pred_path = os.path.join(pred_root, gt_name)
            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
            if gt is None or pred is None:
                print(f'[WARN] skip {gt_name}: gt or pred is None')
                continue
            if gt.shape != pred.shape:
                cv2.imwrite(pred_path, cv2.resize(pred, gt.shape[::-1]))
                pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

            FM.step(pred=pred, gt=gt)
            WFM.step(pred=pred, gt=gt)
            SM.step(pred=pred, gt=gt)
            EM.step(pred=pred, gt=gt)
            MAE.step(pred=pred, gt=gt)

            if args.BR == 'on':
                BR_gt, BR_pred, area = Borders_Capture(
                    cv2.imread(gt_path), cv2.imread(pred_path), int(args.br_rate)
                )
                BR_MAE.step(pred=BR_pred, gt=BR_gt, area=area)
                BR_wF.step(pred=BR_pred, gt=BR_gt)

        except Exception as e:
            print(f'[ERROR] processing {gt_name} in {dataset}: {e}')
            continue

    fm = FM.get_results()[0]['fm']
    wfm = WFM.get_results()['wfm']
    sm = SM.get_results()['sm']
    em = EM.get_results()['em']
    mae = MAE.get_results()['mae']
    fnr = FM.get_results()[1]
    model_r = str(args.model)
    Smeasure_r = str(sm.round(3))
    Wmeasure_r = str(wfm.round(3))
    MAE_r = str(mae.round(3))
    adpEm_r = str(em['adp'].round(3))
    meanEm_r = str('-' if em['curve'] is None else em['curve'].mean().round(3))
    maxEm_r = str('-' if em['curve'] is None else em['curve'].max().round(3))
    adpFm_r = str(fm['adp'].round(3))
    meanFm_r = str(fm['curve'].mean().round(3))
    maxFm_r = str(fm['curve'].max().round(3))
    fnr_r = str(fnr.round(3))

    if args.BR == 'on':
        BRmae = BR_MAE.get_results()['mae']
        BRmae_r = str(BRmae.round(3))
        BRwF = BR_wF.get_results()['wfm']
        BRwF_r = str(BRwF.round(3))
        eval_record = (
            f'Model:{model_r},Dataset:{dataset}||'
            f'Smeasure:{Smeasure_r}; meanEm:{meanEm_r}; wFmeasure:{Wmeasure_r}; MAE:{MAE_r}; '
            f'fnr:{fnr_r}; adpEm:{adpEm_r}; meanEm:{meanEm_r}; maxEm:{maxEm_r}; '
            f'adpFm:{adpFm_r}; meanFm:{meanFm_r}; maxFm:{maxFm_r}; '
            f'BR{args.br_rate}_mae:{BRmae_r}; BR{args.br_rate}_wF:{BRwF_r}'
        )
    else:
        eval_record = (
            f'Model:{model_r},Dataset:{dataset}||'
            f'Smeasure:{Smeasure_r}; meanEm:{meanEm_r}; wFmeasure:{Wmeasure_r}; MAE:{MAE_r}; '
            f'fnr:{fnr_r}; adpEm:{adpEm_r}; meanEm:{meanEm_r}; maxEm:{maxEm_r}; '
            f'adpFm:{adpFm_r}; meanFm:{meanFm_r}; maxFm:{maxFm_r}'
        )

    print(eval_record)
    print('#' * 50)

    os.makedirs('output', exist_ok=True)
    if args.record_path is not None:
        txt = args.record_path
    else:
        txt = 'output/eval_record.txt'  # ← 统一文件名，不带 dataset
    os.makedirs(os.path.dirname(txt), exist_ok=True)
    with open(txt, 'a', encoding='utf-8') as f:
        f.write(eval_record + '\n')

    print(f'[INFO] wrote results to {txt}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='CamoFormer')
    parser.add_argument("--pred_root", default='Prediction/CamoFormer')
    parser.add_argument("--GT_root", default='Dataset/TestData')
    parser.add_argument("--record_path", default=None)
    parser.add_argument("--BR", default='off')
    parser.add_argument("--br_rate", default=15)
    args = parser.parse_args()

    datasets = ['NC4K', 'COD10K', 'CAMO', 'CHAMELEON']
    existed_pred = sorted(os.listdir(args.pred_root))
    print(f'[INFO] found pred subdirs: {existed_pred}')

    for dataset in datasets:
        if dataset in existed_pred:
            try:
                eval(args, dataset)
            except Exception as e:
                print(f'[ERROR] dataset {dataset} failed: {e}')
        else:
            print(f'[INFO] skip {dataset} (not found in {args.pred_root})')

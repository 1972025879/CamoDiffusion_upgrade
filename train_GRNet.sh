accelerate launch \
  --main_process_port=29501 \
  train.py \
  --results_folder ./results_GRNet\
  --config config/camoDiffusion_352x352_GRNet.yaml \
  --num_epoch=150 \
  --batch_size=32 \
  --gradient_accumulate_every=1
accelerate launch --main_process_port 29501 train.py \
    --results_folder ./results_GRNet\
    --config config/camoDiffusion_384x384_GRNet.yaml \
    --num_epoch=20 \
    --batch_size=28 \
    --gradient_accumulate_every=1 \
    --pretrained results_GRNet/model-best.pt \
    --lr_min=0 \
    --set optimizer.params.lr=1e-5
accelerate launch --main_process_port=29501 sample.py \
  --custom_header "CamoDiffusion N/A_GRNet_New_step10" \
  --config config/camoDiffusion_384x384_GRNet.yaml \
  --results_folder /home/4T/wuhao_zjc/aa_new_document/CamoDiffusion/results \
  --checkpoint /home/4T/wuhao_zjc/aa_new_document/CamoDiffusion/results_GRNet/model-best.pt \
  --num_sample_steps 10 \
  --target_dataset CAMO COD10K CHAMELEON NC4K\
  --time_ensemble
# accelerate launch \
#   --main_process_port=29504 \
#   train.py \
#   --config config/camoDiffusion_352x352.yaml \
#   --results_folder ./results\
#   --num_epoch=150 \
#   --batch_size=32 \
#   --gradient_accumulate_every=1
# accelerate launch --main_process_port 29504 train.py \
#   --results_folder ./results\
#   --config config/camoDiffusion_384x384_GRNet.yaml \
#   --num_epoch=20 \
#   --batch_size=28 \
#   --gradient_accumulate_every=1 \
#   --pretrained results/model-best.pt \
#   --lr_min=0 \
#   --set optimizer.params.lr=1e-5
accelerate launch --main_process_port=29504 sample.py \
  --custom_header "CamoDiffusion N/A_step10" \
  --config config/camoDiffusion_384x384.yaml \
  --results_folder /home/4T/wuhao_zjc/aa_new_document/CamoDiffusion/results \
  --checkpoint /home/4T/wuhao_zjc/aa_new_document/CamoDiffusion/results/model-best.pt \
  --num_sample_steps 10 \
  --target_dataset CAMO COD10K CHAMELEON NC4K\
  --time_ensemble
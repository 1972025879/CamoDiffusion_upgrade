accelerate launch \
  --main_process_port=29502 \
  trainORSSD.py \
  --results_folder ./results_ORSSD\
  --config config/SOD/camoDiffusion_352x352.yaml \
  --num_epoch=150 \
  --batch_size=32 \
  --gradient_accumulate_every=1
accelerate launch --main_process_port 29502 trainORSSD.py \
    --results_folder ./results_ORSSD\
    --config config/SOD/camoDiffusion_384x384.yaml \
    --num_epoch=20 \
    --batch_size=28 \
    --gradient_accumulate_every=1 \
    --pretrained results_ORSSD/model-best.pt \
    --lr_min=0 \
    --set optimizer.params.lr=1e-5
accelerate launch --main_process_port=29502 sampleORSSD.py \
  --custom_header "CamoDiffusion N/A_ORSSD_step10" \
  --config config/SOD/camoDiffusion_384x384.yaml \
  --results_folder /home/4T/wuhao_zjc/aa_new_document/CamoDiffusion/results \
  --checkpoint /home/4T/wuhao_zjc/aa_new_document/CamoDiffusion/results_ORSSD/model-best.pt \
  --num_sample_steps 10 \
  --target_dataset ORSSD\
  --time_ensemble
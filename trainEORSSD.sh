accelerate launch \
  --main_process_port=29502 \
  trainEORSSD.py \
  --results_folder ./results_EORSSD\
  --config config/SOD/camoDiffusion_352x352.yaml \
  --num_epoch=150 \
  --batch_size=32 \
  --gradient_accumulate_every=1
accelerate launch --main_process_port 29502 trainEORSSD.py \
    --results_folder ./results_EORSSD\
    --config config/SOD/camoDiffusion_384x384.yaml \
    --num_epoch=20 \
    --batch_size=28 \
    --gradient_accumulate_every=1 \
    --pretrained results_EORSSD/model-best.pt \
    --lr_min=0 \
    --set optimizer.params.lr=1e-5
accelerate launch --main_process_port=29502 sampleEORSSD.py \
  --custom_header "CamoDiffusion N/A_EORSSD_step10" \
  --config config/SOD/camoDiffusion_384x384.yaml \
  --results_folder /home/4T/wuhao_zjc/aa_new_document/CamoDiffusion/results \
  --checkpoint /home/4T/wuhao_zjc/aa_new_document/CamoDiffusion/results_EORSSD/model-best.pt \
  --num_sample_steps 10 \
  --target_dataset EORSSD\
  --time_ensemble
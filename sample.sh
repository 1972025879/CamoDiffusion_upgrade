accelerate launch --main_process_port=29504 sample.py \
  --config config/camoDiffusion_384x384.yaml \
  --results_folder /home/4T/wuhao_zjc/aa_new_document/CamoDiffusion/results \
  --checkpoint /home/4T/wuhao_zjc/aa_new_document/CamoDiffusion/results/model-best.pt \
  --num_sample_steps 10 \
  --target_dataset CAMO COD10K CHAMELEON NC4K\
  --time_ensemble
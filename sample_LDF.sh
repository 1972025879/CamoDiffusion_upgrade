accelerate launch --main_process_port=29510 sample.py \
  --config config/camoDiffusion_384x384_LDF.yaml \
  --results_folder /home/4T/wuhao_zjc/aa_new_document/CamoDiffusion/results \
  --checkpoint /home/4T/wuhao_zjc/aa_new_document/CamoDiffusion/results_LDF/model-best.pt \
  --num_sample_steps 10 \
  --target_dataset CAMO COD10K CHAMELEON NC4K\
  --time_ensemble

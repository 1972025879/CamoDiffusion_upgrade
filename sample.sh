accelerate launch --main_process_port=29510 sample.py \
  --config config/camoDiffusion_384x384.yaml \
  --results_folder /home/4T/wuhao_zjc/aa_new_document/CamoDiffusion/results \
  --checkpoint /home/4T/wuhao_zjc/aa_new_document/CamoDiffusion_LDF/results/model-best.pt \
  --num_sample_steps 10 \
  --target_dataset CAMO\
  --time_ensemble
accelerate launch --main_process_port=29510 sample.py \
  --config config/camoDiffusion_384x384.yaml \
  --results_folder /home/4T/wuhao_zjc/aa_new_document/CamoDiffusion/results \
  --checkpoint /home/4T/wuhao_zjc/aa_new_document/CamoDiffusion_LDF/results/model-best.pt \
  --num_sample_steps 10 \
  --target_dataset CHAMELEON\
  --time_ensemble
accelerate launch --main_process_port=29510 sample.py \
  --config config/camoDiffusion_384x384.yaml \
  --results_folder /home/4T/wuhao_zjc/aa_new_document/CamoDiffusion/results \
  --checkpoint /home/4T/wuhao_zjc/aa_new_document/CamoDiffusion_LDF/results/model-best.pt \
  --num_sample_steps 10 \
  --target_dataset COD10K\
  --time_ensemble
accelerate launch --main_process_port=29510 sample.py \
  --config config/camoDiffusion_384x384.yaml \
  --results_folder /home/4T/wuhao_zjc/aa_new_document/CamoDiffusion/results \
  --checkpoint /home/4T/wuhao_zjc/aa_new_document/CamoDiffusion_LDF/results/model-best.pt \
  --num_sample_steps 10 \
  --target_dataset NC4K\
  --time_ensemble
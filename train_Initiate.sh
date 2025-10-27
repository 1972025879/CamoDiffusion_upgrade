accelerate launch \
  --main_process_port=29503 \
  train.py \
  --config config/camoDiffusion_352x352_Initiate.yaml \
  --results_folder ./results_Initiate\
  --num_epoch=150 \
  --batch_size=8 \
  --gradient_accumulate_every=1
accelerate launch \
  --main_process_port=29503 \
  train.py \
  --results_folder ./results_LDF\
  --config config/camoDiffusion_352x352_LDF.yaml \
  --num_epoch=150 \
  --batch_size=32 \
  --gradient_accumulate_every=1
accelerate launch \
  --main_process_port=29504 \
  train.py \
  --config config/camoDiffusion_352x352.yaml \
  --results_folder ./results\
  --num_epoch=150 \
  --batch_size=32 \
  --gradient_accumulate_every=1
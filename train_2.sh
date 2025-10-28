accelerate launch --main_process_port 29504 train.py \
    --results_folder ./results\
    --config config/camoDiffusion_384x384.yaml \
    --num_epoch=20 \
    --batch_size=28 \
    --gradient_accumulate_every=1 \
    --pretrained results/model-best.pt \
    --lr_min=0 \
    --set optimizer.params.lr=1e-5

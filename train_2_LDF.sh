accelerate launch --main_process_port 29512 train.py \
    --results_folder ./results_LDF\
    --config config/camoDiffusion_384x384_LDF.yaml \
    --num_epoch=20 \
    --batch_size=28 \
    --gradient_accumulate_every=1 \
    --pretrained results_LDF/model-best.pt \
    --lr_min=0 \
    --set optimizer.params.lr=1e-5

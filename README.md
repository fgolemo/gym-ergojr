Install via

    pip install -e .
    
train with `https://github.com/fgolemo/pytorch-a2c-ppo-acktr`

with (over in the pytorch-a2c-ppo... directory):

    python3 main.py \
    --env-name "ErgoReacher-Headless-Simple-v1" \
    --custom-gym gym_ergojr \
    --num-processes 1 \
    --vis-host "http://localhost" \
    --vis-port 8097 \
    --algo ppo \
    --use-gae \
    --vis-interval 1 \
    --log-interval 1 \
    --num-stack 1 \
    --num-steps 2048 \
    --lr 3e-4 \
    --entropy-coef 0 \
    --ppo-epoch 10 \
    --num-mini-batch 32 \
    --gamma 0.99 \
    --tau 0.95 \
    --save-interval 10 \
    --num-frames 200000 \
    --seed 1
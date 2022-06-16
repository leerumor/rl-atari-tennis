# RL-Atari-Tennis

The baseline dqn code is copied from [cleanrl](https://github.com/vwxyzjn/cleanrl)

## Requirements

```
python=3.8

pip install -r requirements.txt
```

## Train

```
sh train.sh
```

The best model was trained by two-stage.

Key parameters are (others remain the same):

```
# first stage
--total-timesteps 2000000 \
--learning-rate 0.0001 \
--start-e 1 \
--end-e 0.02 \
--exploration-fraction 0.2 \

# second stage
--total-timesteps 10000000 \
--learning-rate 0.00002 \
--start-e 0.1 \
--end-e 0.02 \
--exploration-fraction 0.04 \
```

## Play

```
python play.py
```

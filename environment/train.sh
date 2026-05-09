python scripts/train_recurrent_gat.py \
  --timesteps 500000 \
  --preset full \
  --n-envs 8 \
  --eval-episodes 30 

python scripts/train_recurrent_gat.py \
  --timesteps 500000 \
  --preset baseline \
  --n-envs 8 \
  --eval-episodes 30 

python scripts/train_recurrent_gat.py \
  --timesteps 500000 \
  --preset no_legitimacy \
  --n-envs 8 \
  --eval-episodes 30 

python scripts/train_recurrent_gat.py \
  --timesteps 500000 \
  --preset no_occupation_cost \
  --n-envs 8 \
  --eval-episodes 30 

python scripts/train_recurrent_gat.py \
  --timesteps 500000 \
  --preset no_neutral_posture \
  --n-envs 8 \
  --eval-episodes 30 


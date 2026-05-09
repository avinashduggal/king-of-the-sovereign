python scripts/train_ppo_gnn.py \
  --total-timesteps 500000 \
  --preset no_legitimacy \
  --n-envs 8 \
  --eval-episodes 30 

python scripts/train_ppo_gnn.py \
  --total-timesteps 500000 \
  --preset no_occupation_cost \
  --n-envs 8 \
  --eval-episodes 30 

python scripts/train_ppo_gnn.py \
  --total-timesteps 500000 \
  --preset no_neutral_posture \
  --n-envs 8 \
  --eval-episodes 30 


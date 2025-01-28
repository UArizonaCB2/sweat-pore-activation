chmod +x ./scripts/train.sh # Excute permission

# The user can add --tag 'info' for naming the ML model (defualt is None)
# python train.py --batchSize 8 --CNN 'CNN4LayersV4_p32' --epochs 25 --device 'mps'
python train.py --batchSize 8 --CNN 'CNN4LayersV4_p32' --epochs 25 --device 'mps'
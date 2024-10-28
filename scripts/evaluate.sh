chmod +x ./scripts/evaluate.sh # Excute permission

patchesDir="Preprocessing/centralizedPatches/32X32/"
python evaluate.py --batchSize 8 --CNNmodel 'CNN4Layers_p32_e20_135bmps.model' --prediction '6bmp' --device 'mps' --patchesDir $patchesDir
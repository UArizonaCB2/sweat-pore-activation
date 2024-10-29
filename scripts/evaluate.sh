#!/bin/bash
chmod +x ./scripts/evaluate.sh # Excute permission

# Golbal variables
centralized=true  # true: centralized pore pathces, false: sliding window pathces
imgName="2" #ex. 2.bmp ---> 2

if [ "$centralized" = true ]; then
    patchesDir="Preprocessing/centralizedPatches/32X32/"

    # 1. Run the preprocessing step
    echo "Running preprocessing to create prediction on the unseen centralized pore patches..."
    rm -rf Preprocessing/centralizedPatches/32X32/* #Delete the existing patches 
    python Preprocessing/centralize_sweatpores.py --imgName $imgName
    # 2. Run the evaluation
    echo "Starting evaluation..."
    python evaluate.py --batchSize 8 --CNNmodel 'CNN4Layers_p32_e20_135bmps.model' --prediction $imgName'bmp' --device 'mps' --patchesDir $patchesDir
else
    echo "Running preprocessing to create prediction on the unseen sliding window patches..."
    patchesDir="Preprocessing/testingModel_output_patches/${imgName}bmp/patch_size/32X32/"

    #Run the evaluation
    echo "Starting evaluation..."
    python evaluate.py --batchSize 8 --CNNmodel 'CNN4Layers_p32_e20_135bmps.model' --prediction $imgName'bmp' --device 'mps' --patchesDir $patchesDir
fi
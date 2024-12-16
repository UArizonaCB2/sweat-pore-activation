#!/bin/bash
chmod +x ./analyzeSweatPores.sh # Execute permission

# # 1. Run the processVideo.py to fetch frames
# echo "Processing the Video..."
# python processVideo.py

# # 2. Clear out the prior results in prediction.txt before processing
# echo "Clearing out the prior prediction..."
# > prediction.txt

# # 3. Iterate through each frame 
# for frame in frames/videoToframes/*.bmp
# do
#     # 3.1 Create dataset for current frame 
#     basename=$(basename "$frame" .bmp)
#     echo "Creating Dataset for $basename ..."
#     python ./dataset/slidingWindow/createPatches.py --frameName "$basename"
    
#     # 3.2 Evaluate the dataset and store the prediction
#     echo "Evaluating Frame $basename ..."
#     patchesDir="./VideoProcessing/dataset/slidingWindow/32X32/" 
#     model="CNN4Layers_p32_e20_135bmps.model"
#     cd ..
#     python VideoProcessing/prediction.py --batchSize 8 --CNNmodel $model --device 'mps' --patchesDir $patchesDir --currentFrameNumber $frame
#     cd VideoProcessing
# done

# 4. Make a scatterplot and Create a video
python analysis.py

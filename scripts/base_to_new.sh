#!/bin/bash
DATA=/path/to/data
TRAINER=TAP
SHOTS=16

DATASET_SETTINGS=(
    "dtd,1.0,0.002,0.006,3.0"
    "eurosat,1.0,0.004,0.004,1.5"
    "oxford_pets,1.0,0.004,0.004,1.5"
    "food101,1.0,0.004,0.004,1.5"
    "oxford_flowers,1.0,0.002,0.006,3.0"
    "stanford_cars,0.95,0.002,0.006,3.0"
    "fgvc_aircraft,1.0,0.004,0.004,1.5"
    "ucf101,0.95,0.002,0.006,3.0"
    "caltech101,1.0,0.002,0.006,3.0"
    "sun397,0.95,0.004,0.004,1.5"
    "imagenet,0.95,0.004,0.004,1.5"
)
vision_num_epochs=50
text_num_epochs=10

bs=32
OPTIM_NAME=adamw

for SEED in 1 2 3; do
    for DATASET_SETTING in "${DATASET_SETTINGS[@]}"; do
        IFS=',' read -r DATASET quantile text_lr vision_lr u3 <<< "$DATASET_SETTING"
        TRAIN_OUT_DIR=output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/seed${SEED}
        if [ -f "${TRAIN_OUT_DIR}/vision_prompt_learner/model.pth.tar-50" ]; then
            echo "Oops! The results exist at ${TRAIN_OUT_DIR} (so skip this job)"
        else
            python train.py \
                    --root ${DATA} \
                    --seed ${SEED} \
                    --trainer ${TRAINER} \
                    --dataset-config-file configs/datasets/${DATASET}.yaml \
                    --config-file configs/trainers/${TRAINER}/TAP.yaml \
                    --output-dir ${TRAIN_OUT_DIR} \
                    OPTIM.NAME ${OPTIM_NAME} \
                    OPTIM.VISION_LR ${vision_lr} \
                    OPTIM.TEXT_LR ${text_lr} \
                    DATASET.NUM_SHOTS 16 \
                    DATASET.SUBSAMPLE_CLASSES base \
                    DATALOADER.TRAIN_X.BATCH_SIZE ${bs} \
                    MODEL.TAP.U3 ${u3} \
                    MODEL.TAP.MAX_LEN_QUANTILE ${quantile}
        
        TEST_OUT_DIR=output/base2new/test_new/${DATASET}/shots_${SHOTS}/${TRAINER}/seed${SEED}
        python train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${TRAINER}/TAP.yaml \
            --output-dir ${TEST_OUT_DIR} \
            --model-dir ${TRAIN_OUT_DIR} \
            --load-epoch ${num_epochs} \
            --eval-only \
            DATASET.SUBSAMPLE_CLASSES new \
            DATALOADER.TRAIN_X.BATCH_SIZE ${bs} \
            OPTIM.NAME ${OPTIM_NAME} \
            OPTIM.VISION_LR ${vision_lr} \
            OPTIM.TEXT_LR ${text_lr} \
            MODEL.TAP.U3 ${u3} \
            MODEL.TAP.MAX_LEN_QUANTILE ${quantile}
    done
done
#! /bin/bash

print_info_msg ()
{
  local msg="$1"

  echo -e '\033[1;36m'$msg'\033[0m'
}

print_fail_msg ()
{
  local msg="$1"

  echo -e '\033[1;31m'$msg'\033[0m'
}

check_success ()
{
  local n_args=$#
  local status=$1

  if [ $status -ne 0 ]; then
    echo '\033[1;31m'"Failure..."'\033[0m'
    exit
  fi
}


# repeat all for simulation and real
SIM=(
  ""  # real
  "_sim"
)

SKIP_AUG=0
SKIP_TRAIN=0
SKIP_CALC_RMSE=0
SKIP_COMPARE=0

for sim in "${SIM[@]}" ; do

  # define the models to train
  MODELS=(
          "../logs/models/rn18_dmp$sim.bin:resnet18_dmp" 
          # "../logs/models/rn50_dmp$sim.bin:resnet50_dmp" 
          "../logs/models/vimednet$sim.bin:vimednet"
      )

  MAIN_PATH="../logs/unveil_demo_dataset$sim/"

  TRAIN_SET=$MAIN_PATH'train/'
  DEV_SET=$MAIN_PATH'dev/'
  TEST_SET=$MAIN_PATH'test/'

  # ========  Augment datasets ========
  if ! (( $SKIP_AUG != 0 )); then

    N_AUG=100
    SEED=0
    VIZ=0

    # Both single and multi grapebunch datasets undergo the same augmentation except for translation.
    # For multi grapebunches will apply small translations so that the target grapebunch remains the one closer to the center of the image.

    # single grapebunch dataset
    TRAIN_1_SET=$MAIN_PATH'single_grapebunch/train/'
    DEV_1_SET=$MAIN_PATH'single_grapebunch/dev/'
    TEST_1_SET=$MAIN_PATH'single_grapebunch/test/'
    Single_TFs='../config/single_grape_augment_tfs.txt'

    print_info_msg "Augmenting single grapenucn TRAIN-set..."
    python3 augment_dataset.py --iters=$N_AUG --dataset=$TRAIN_1_SET --save_to=$TRAIN_SET --seed=$SEED --transforms=$Single_TFs --append=0 --viz=$VIZ
    check_success $?

    print_info_msg "Augmenting single grapenucn DEV-set..."
    python3 augment_dataset.py --iters=$N_AUG --dataset=$DEV_1_SET --save_to=$DEV_SET --seed=$SEED --transforms=$Single_TFs --append=0 --viz=$VIZ
    check_success $?

    print_info_msg "Augmenting single grapenucn TEST-set..."
    python3 augment_dataset.py --iters=$N_AUG --dataset=$TEST_1_SET --save_to=$TEST_SET --seed=$SEED --transforms=$Single_TFs --append=0 --viz=$VIZ
    check_success $?

    # multi grapebunch dataset
    TRAIN_m_SET=$MAIN_PATH'multi_grapebunch/train/'
    DEV_m_SET=$MAIN_PATH'multi_grapebunch/dev/'
    TEST_m_SET=$MAIN_PATH'multi_grapebunch/test/'
    Multi_TFs='../config/multi_grape_augment_tfs.txt'

    print_info_msg "Augmenting multi grapenucn TRAIN-set..."
    python3 augment_dataset.py --iters=$N_AUG --dataset=$TRAIN_m_SET --save_to=$TRAIN_SET --seed=$SEED --transforms=$Multi_TFs --append=1 --viz=$VIZ
    check_success $?

    print_info_msg "Augmenting multi grapenucn DEV-set..."
    python3 augment_dataset.py --iters=$N_AUG --dataset=$DEV_m_SET --save_to=$DEV_SET --seed=$SEED --transforms=$Multi_TFs --append=1 --viz=$VIZ
    check_success $?

    print_info_msg "Augmenting multi grapenucn TEST-set..."
    python3 augment_dataset.py --iters=$N_AUG --dataset=$TEST_m_SET --save_to=$TEST_SET --seed=$SEED --transforms=$Multi_TFs --append=1 --viz=$VIZ
    check_success $?
  
  fi

  # ========  Train models ========

  if ! (( $SKIP_TRAIN != 0 )); then

    EPOCHS=150
    BATCH_SIZE=32
    TRAIN_SEED=68213

    for model_name_type in "${MODELS[@]}" ; do
        model_name="${model_name_type%%:*}"
        model_type="${model_name_type##*:}"

        print_info_msg "Training '"$model_name' : '$model_type"' ..."
        python3 train_model.py --model=$model_type --save_as=$model_name --train_set=$TRAIN_SET --dev_set=$DEV_SET --test_set=$TEST_SET --epochs=$EPOCHS --batch_size=$BATCH_SIZE --seed=$TRAIN_SEED
        check_success $?    
    done

  fi

  # ========  Evaluate models ========
  if ! (( $SKIP_CALC_RMSE != 0 )); then

    for model_name_type in "${MODELS[@]}" ; do
      model_name="${model_name_type%%:*}"

      print_info_msg "Calculating RMSE for "$model_name"..."
      python3 calc_mse.py --model=$model_name --datasets $TRAIN_SET $DEV_SET $TEST_SET
      check_success $?    
    done

  fi
  
  # aggrate all models in a string separated by " " and then compare them
  if ! (( $SKIP_COMPARE != 0 )); then
    COMPARE_SET=$TEST_SET
    models=""
    for model_name_type in "${MODELS[@]}" ; do
        model_name="${model_name_type%%:*}"
        models=$models" "$model_name
    done

    print_info_msg "Comparing models..."
    python3 compare_models.py --models $models --dataset=$COMPARE_SET --shuffle=1 --batch_size=4
  fi

done
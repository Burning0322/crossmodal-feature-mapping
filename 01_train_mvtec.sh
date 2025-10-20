export CUDA_VISIBLE_DEVICES=0,1

epochs=200
batch_size=2

class_names=("bagel")

for class_name in "${class_names[@]}"
    do
        python cfm_training.py --class_name $class_name --epochs_no $epochs --batch_size $batch_size 
    done
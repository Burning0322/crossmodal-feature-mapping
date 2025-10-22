export CUDA_VISIBLE_DEVICES=0,1

epochs=50
batch_size=2

#"bagel" "cable_gland" "carrot" "cookie" "dowel" "foam" "peach" "potato" "rope" "tire"
class_names=("potato")

for class_name in "${class_names[@]}"
    do
        python cfm_training.py --class_name $class_name --epochs_no $epochs --batch_size $batch_size 
    done
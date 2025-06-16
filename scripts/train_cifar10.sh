today=`date +%T`
echo $today
gpu_id=$1
trainer=$2

if [[ "${trainer}" == *"supervised"* ]] || [[ "${trainer}" == *"mixmatch"* ]] || [[ "${trainer}" == *"fixmatch"* ]] || \
   [[ "${trainer}" == *"comatch"* ]]    || [[ "${trainer}" == *"simmatch"* ]] || [[ "${trainer}" == *"softmatch"* ]]; then
    ssl_type=classic_cv

elif [[ "${trainer}" == *"mtc"* ]] || [[ "${trainer}" == *"openmatch"* ]] || [[ "${trainer}" == *"safestudent"* ]] || \
     [[ "${trainer}" == *"iomatch"* ]] || [[ "${trainer}" == *"uagreg"* ]] || [[ "${trainer}" == *"dac"* ]]; then
    ssl_type=openset_cv
fi

# CIFAR10_TRAIN
dataset=cifar10
num_classes=6
for seed in 1 2 3
do
    for n_labels in 5 10 25
    do
        num_labels=$((n_labels * num_classes))
        CUDA_VISIBLE_DEVICES=$gpu_id python train.py --c config/${ssl_type}/${trainer}/${trainer}_${dataset}_${num_classes}_${num_labels}_${seed}.yaml
    done
done

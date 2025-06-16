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
dataset=in30
for seed in 1 2 3
do
    for percent in p1 p5
    do
        CUDA_VISIBLE_DEVICES=$gpu_id python train.py --c config/openset_cv/${trainer}/${trainer}_in30_${percent}_${seed}.yaml
    done
done

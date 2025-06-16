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

# SEMI_INAT_2021_TRAIN
dataset=semi_inat
for seed in 1 2 3
do
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py --c config/${ssl_type}/${trainer}/${trainer}_semi_inat_${seed}.yaml
done

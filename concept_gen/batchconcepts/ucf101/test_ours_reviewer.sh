#####
# ablation of concept
########################################################
export PYTHONPATH=/hdd2/lh/project:$PYTHONPATH
mkdir -p results
# test_sets="SUN397/Aircraft/eurosat/Cars/Food101/Pets/Flower102/Caltech101/DTD/UCF101/I"
test_sets="eurosat/DTD/Aircraft/Pets/UCF101"
# test_sets="I"




# CUDA_VISIBLE_DEVICES=1 python zero_shot_hc_infer_nowanbd_batch_unique_diversity.py \
#     --test_sets $test_sets \
#     --sample_mode multiple \
#     --combine_op or \
#     --len_prompts 3 \
#     --al_mode concept_mad_noise  \
#     --tau 1.0 \
#     --lambda_threshold 2.5 \
#     --sampling_times 50 \
#     --decompose_or_prompts True \
#     --num_runs 3 \
#     --sampling_method  dpp \
#     --max_combinations 500 \
#     --resolution 224 \
#     --result_path results_report/results_reviewer.json &


# CUDA_VISIBLE_DEVICES=1 python zero_shot_hc_infer_nowanbd_batch_unique_diversity.py \
#     --test_sets $test_sets \
#     --sample_mode multiple \
#     --combine_op or \
#     --len_prompts 3 \
#     --al_mode concept_avg  \
#     --tau 1.0 \
#     --lambda_threshold 2.5 \
#     --sampling_times 50 \
#     --decompose_or_prompts True \
#     --num_runs 3 \
#     --sampling_method  dpp \
#     --max_combinations 500 \
#     --resolution 224 \
#     --result_path results_report/results_reviewer.json &



# CUDA_VISIBLE_DEVICES=1 python zero_shot_hc_infer_nowanbd_batch_unique_diversity.py \
#     --test_sets $test_sets \
#     --sample_mode multiple \
#     --combine_op or \
#     --len_prompts 3 \
#     --al_mode concept_mad_noise  \
#     --tau 1.0 \
#     --lambda_threshold 2.5 \
#     --sampling_times 50 \
#     --decompose_or_prompts False \
#     --num_runs 3 \
#     --concept_type 50_flash2.5 \
#     --sampling_method  dpp \
#     --max_combinations 500 \
#     --resolution 224 \
#     --result_path results_report/results_reviewer.json &


CUDA_VISIBLE_DEVICES=2 python zero_shot_hc_infer_nowanbd_batch_unique_diversity.py \
    --test_sets $test_sets \
    --sample_mode multiple \
    --combine_op or \
    --len_prompts 3 \
    --al_mode concept_mad_noise  \
    --tau 1.0 \
    --lambda_threshold 2.5 \
    --sampling_times 50 \
    --decompose_or_prompts False \
    --num_runs 3 \
    --concept_type 50_2.0flash \
    --sampling_method  dpp \
    --max_combinations 500 \
    --resolution 224 \
    --result_path results_report/results_reviewer.json &


CUDA_VISIBLE_DEVICES=3 python zero_shot_hc_infer_nowanbd_batch_unique_diversity.py \
    --test_sets $test_sets \
    --sample_mode multiple \
    --combine_op or \
    --len_prompts 3 \
    --al_mode concept_mad_noise  \
    --tau 1.0 \
    --lambda_threshold 2.5 \
    --sampling_times 50 \
    --decompose_or_prompts False \
    --num_runs 3 \
    --concept_type 50_flash2.5lite \
    --sampling_method  dpp \
    --max_combinations 500 \
    --resolution 224 \
    --result_path results_report/results_reviewer.json &
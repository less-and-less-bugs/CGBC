#####
# ablation of concept
########################################################
export PYTHONPATH=/hdd2/lh/project:$PYTHONPATH
mkdir -p results
test_sets="SUN397/Aircraft/eurosat/Cars/Food101/Pets/Flower102/Caltech101/DTD/UCF101"
# test_sets="DTD/Aircraft/Pets/UCF101"
# test_sets="I/A/R/K/V"


CUDA_VISIBLE_DEVICES=1 python zero_shot_hc_infer_nowanbd_batch_unique_diversity.py \
    --test_sets $test_sets \
    --sample_mode multiple \
    --combine_op or \
    --len_prompts 3 \
    --al_mode concept_mad_noise  \
    --tau 1.0 \
    --lambda_threshold 2.5 \
    --sampling_times 50 \
    --num_runs 3 \
    --sampling_method  dpp \
    --concept_type 50_sim \
    --max_combinations 500 \
    --resolution 224 \
    --result_path results_report/results_concept.json &



CUDA_VISIBLE_DEVICES=1 python zero_shot_hc_infer_nowanbd_batch_unique_diversity.py \
    --test_sets $test_sets \
    --sample_mode multiple \
    --combine_op or \
    --len_prompts 3 \
    --al_mode concept_mad_noise  \
    --tau 1.0 \
    --lambda_threshold 2.5 \
    --sampling_times 16 \
    --num_runs 3 \
    --sampling_method  dpp \
    --concept_type 50_sim \
    --max_combinations 500 \
    --resolution 224 \
    --result_path results_report/results_concept.json &
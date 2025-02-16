gpu=0
CUDA_VISIBLE_DEVICES=$gpu python ./scripts/train_text_generation.py \
    --base_path_to_store_results ./rl4lms_exps \
    --project_name summarization_with_hint \
    --experiment_name bart-large-cnn_nlpo_on_supervised-general_1024_64 \
    --config_path ./scripts/training/task_configs_general/summarization_with_hint/flan-t5_nlpo_on_supervised-general.yml





gpu=0
CUDA_VISIBLE_DEVICES=$gpu python ./scripts/train_text_generation.py \
    --base_path_to_store_results ./rl4lms_exps \
    --project_name summarization_with_hint \
    --experiment_name flan-t5-large_nlpo_on_supervised-attack \
    --config_path ./scripts/training/task_configs_attack/summarization_with_hint/flan-t5_nlpo_on_supervised-attack.yml
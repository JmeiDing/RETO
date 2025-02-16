# len(general_train)=810
# len(general_valid)=101
# len(general_test)=103

#--task extraction
#--task summarization
# training
gpu=0
CUDA_VISIBLE_DEVICES=$gpu python -m sft4lms.Summarization.run_sft_t5 --task extraction \
                                                                        --n_train 810 \
                                                                        --n_val 101 \
                                                                        --n_test 103 \
                                                                        --dataset attack \
                                                                        --load_strategy load_initial \
                                                                        --extraction_mode textrank \
                                                                        --extraction_source all \
                                                                        --model ./pretrain_model/flan-t5-large \
                                                                        --max_ctx_len 1024 \
                                                                        --max_tgt_len 64 \
                                                                        --train_batch_size 2 \
                                                                        --eval_batch_size 2 \
                                                                        --learning_rate 2e-5 \
                                                                        --epochs 1 \
                                                                        --logging_steps 2 \
                                                                        --save_total_limit 1 \
                                                                        --early_stopping_patience 5 \
                                                                        --do_inference \
                                                                        --do_train \
                                                                        --length_penalty 2 \
                                                                        --no_repeat_ngram_size 3 \







                                                                        # --push_to_hub



# len(general_train)=1077
# len(general_valid)=135
# len(general_test)=133

#--task extraction
#--task summarization
# training
#gpu=0
python ./sft4lms/Summarization/run_sft_t5_re.py     --task extraction \
                                                    --n_train 1077 \
                                                    --n_val 135 \
                                                    --n_test 133 \
                                                    --dataset general \
                                                    --load_strategy load_initial \
                                                    --extraction_mode textrank \
                                                    --extraction_source all \
                                                    --model ./pretrain_model/flan-t5-large \
                                                    --max_ctx_len 1024 \
                                                    --max_tgt_len 128 \
                                                    --train_batch_size 2 \
                                                    --eval_batch_size 2 \
                                                    --learning_rate 2e-5 \
                                                    --epochs 2 \
                                                    --with_tracking \
                                                    --checkpointing_steps epoch \
                                                    --pad_to_max_length

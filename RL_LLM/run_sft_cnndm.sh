#gpu=0,1,2

#for n_train in 1000 2000 4000
#do
## training
#CUDA_VISIBLE_DEVICES=$gpu python -m sft4lms.Summarization.run_sft_t5 --task extraction \
#                                                                        --n_train $n_train \
#                                                                        --n_val 500 \
#                                                                        --n_test 500 \
#                                                                        --dataset cnndm \
#                                                                        --load_strategy load_initial \
#                                                                        --extraction_mode textrank \
#                                                                        --extraction_source all \
#                                                                        --model flan-t5-large \
#                                                                        --train_batch_size 8 \
#                                                                        --eval_batch_size 24 \
#                                                                        --learning_rate 2e-5 \
#                                                                        --epochs 5 \
#                                                                        --logging_steps 100 \
#                                                                        --save_total_limit 3 \
#                                                                        --early_stopping_patience 5 \
#                                                                        --do_inference \
#                                                                        --do_train \
#                                                                        # --push_to_hub
#done
#--task extraction
#--task summarization
# training
gpu=0
CUDA_VISIBLE_DEVICES=$gpu python -m sft4lms.Summarization.run_sft_t5 --task extraction \
                                                                        --n_train 10 \
                                                                        --n_val 5 \
                                                                        --n_test 5 \
                                                                        --dataset cnndm \
                                                                        --load_strategy load_initial \
                                                                        --extraction_mode textrank \
                                                                        --extraction_source all \
                                                                        --model ./pretrain_model/flan-t5-large \
                                                                        --train_batch_size 2 \
                                                                        --eval_batch_size 2 \
                                                                        --learning_rate 2e-5 \
                                                                        --epochs 1 \
                                                                        --logging_steps 2 \
                                                                        --save_total_limit 1 \
                                                                        --early_stopping_patience 5 \
                                                                        --do_inference \
                                                                        --do_train \
                                                                        # --push_to_hub


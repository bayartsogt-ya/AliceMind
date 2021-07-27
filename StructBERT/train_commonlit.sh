mkdir output
run_classifier_multi_task.py \
    --data_dir data/fold_0 \
    --bert_config_file config/large_bert_config.json \
    --task_name commonlit \
    --vocab_file config/vocab.txt \
    --output_dir output \
    --init_checkpoint models/en_model \
    --max_seq_length 256 \
    --do_train \
    --do_eval \
    --lr_decay_factor 0.1 \
    --dropout 0.1 \
    --train_batch_size 8 \
    --eval_batch_size 8 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --warmup_proportion 0.1 \
    --save_model  \
    --seed 1000 \
    --num_workers 2 \
    --debug

    # --fast_train \

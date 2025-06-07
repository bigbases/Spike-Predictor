nohup python -u train_spikformer.py \
        --seed 42 \
        --batch_size 32 \
        --fine_tune_lr 1e-4 \
        --epochs 30 \
        --depths 6 \
        --max_length 128 \
        --dim 768 \
        --tau 10.0 \
        --common_thr 1.0 \
        --num_step 16 \
        --output_path "saved_models/cnn2m_spikformer" \
        --tokenizer_path "bert-base-cased" \
        --task_name SICK \
        > "train_SICK.log" &
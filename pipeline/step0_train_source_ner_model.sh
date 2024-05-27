SEED=1
EPOCH=10
MAX_LENGTH=128

DATA_DIR="path/to/xlmr/data-bin/"
PRETRAINED_ENCODER="xlm_roberta_base"
PRETRAINED_ENCODER_PATH="path/to/output_dir/${PRETRAINED_ENCODER}/checkpoint-best/"

if [ ${PRETRAINED_ENCODER} == "xlm-roberta-large" ]; then
    BATCH_SIZE=32
    GRAD_ACC=10
    LR=3e-6
	EVALUATE_STEPS=2500
else
    BATCH_SIZE=32
    GRAD_ACC=5
    LR=1e-5
	EVALUATE_STEPS=1000
fi

LANGS="en"
PREDICT_LANGS="en"
OUTPUT_DIR="path/to/output_dir/xlm_roberta_base/"
mkdir -p ${OUTPUT_DIR}

python3.7 xTune/src/run_tag.py --model_type xlmr --model_name_or_path ${PRETRAINED_ENCODER_PATH} \
  --do_train --do_eval --do_predict --do_predict_dev --predict_langs ${PREDICT_LANGS} \
  --train_langs en --data_dir ${DATA_DIR} --labels ${DATA_DIR}labels.txt \
  --per_gpu_train_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${GRAD_ACC} \
  --per_gpu_eval_batch_size 64 --learning_rate ${LR} --num_train_epochs ${EPOCH} \
  --max_seq_length ${MAX_LENGTH} --noised_max_seq_length ${MAX_LENGTH} \
  --output_dir ${OUTPUT_DIR} --overwrite_output_dir --evaluate_during_training --logging_steps 50 \
  --evaluate_steps $EVALUATE_STEPS --seed ${SEED} --warmup_steps -1 \
  --save_only_best_checkpoint --eval_all_checkpoints --eval_patience -1 \
  --fp16 --fp16_opt_level O2 --hidden_dropout_prob 0.1 \
  --original_loss 2>&1 | tee -a ${OUTPUT_DIR}log.txt


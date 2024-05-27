NODES=1
GPUS=1
LR=1e-4
UPDATE_FREQ=8
MAX_TOKENS=256
MAX_EPOCH=10
MAX_UPDATES=4000
SUBWORD_PROB=0.0
SLOT_METHOD="insert"
SLOT_PROB=0.85
MAX_SLOT_NUM=14
OMPI_COMM_WORLD_RANK=${RANK:-0}
MASTER_PORT=2424
MASTER_ADDR=${MASTER_ADDR:-'172.18.224.1'}

TEXT="path/to/langs1/:path/to/langs2/"  # dataset directories, seperated by colons ":"
PRETRAINED_MODEL="path/to/m2m_checkpoint_baseline-003.pt"
SPM_MODEL="path/to/m2m/spm.model"

LANGS="en,ru"
LANG_PAIRS="en-ru,ru-en"

SAVE_DIR="path/to/save_dir/"
mkdir -p ${SAVE_DIR}

echo "Start Training ..."
if [ ! -f ${SAVE_DIR}/checkpoint_last.pt ]; then
  python3.7 -m torch.distributed.run --nproc_per_node=${GPUS} --nnodes=${NODES} --rdzv_backend=static\
    --node_rank=${OMPI_COMM_WORLD_RANK} --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} m2m/train.py ${TEXT} \
    --save-dir ${SAVE_DIR} --restore-file ${PRETRAINED_MODEL} --arch "transformer_wmt_en_de_big" \
    --encoder-normalize-before --decoder-normalize-before --encoder-layers 12 --decoder-layers 12 \
    --task translation_multi_simple_epoch --sampling-method "linear" --sampling-temperature 5.0 \
    --min-sampling-temperature 1.0 --warmup-epoch 5 \
    --encoder-langtok "src" --langtoks '{"main":("src","tgt")}' \
    --langs ${LANGS} --lang-pairs ${LANG_PAIRS} \
    --truncate-source --enable-reservsed-directions-shared-datasets --load-alignments \
    --share-all-embeddings --max-source-positions 1024 --max-target-positions 1024 \
    --criterion "label_smoothed_cross_entropy" --label-smoothing 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr-scheduler inverse_sqrt --lr ${LR} --warmup-init-lr 1e-07 --stop-min-lr 1e-07 \
    --warmup-updates 4000 --max-update ${MAX_UPDATES} --max-epoch ${MAX_EPOCH} \
    --attention-dropout 0.1 --dropout 0.1 --max-tokens ${MAX_TOKENS} --update-freq ${UPDATE_FREQ} \
    --seed 1 --log-format simple --skip-invalid-size-inputs-valid-test --fp16 \
    --bpe sentencepiece --sentencepiece-model ${SPM_MODEL} --subword-prob ${SUBWORD_PROB} \
    --max-slot-num ${MAX_SLOT_NUM} --slot-prob ${SLOT_PROB} --slot-method ${SLOT_METHOD} \
    --reset-optimizer --reset-meters --reset-lr-scheduler --reset-dataloader \
    --ddp-backend=no_c10d 2>&1 | tee -a ${SAVE_DIR}/train.log
else
  python3.7 -m torch.distributed.run --nproc_per_node=${GPUS} --nnodes=${NODES} --rdzv_backend=static\
    --node_rank=${OMPI_COMM_WORLD_RANK} --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} train.py ${TEXT} \
    --save-dir ${SAVE_DIR} --arch "transformer_wmt_en_de_big" \
    --encoder-normalize-before --decoder-normalize-before --encoder-layers 12 --decoder-layers 12 \
    --task translation_multi_simple_epoch --sampling-method "linear" --sampling-temperature 5.0 \
    --min-sampling-temperature 1.0 --warmup-epoch 5 
    --encoder-langtok "src" --langtoks '{"main":("src","tgt")}' \
    --langs ${LANGS} --lang-pairs ${LANG_PAIRS} \
    --truncate-source --enable-reservsed-directions-shared-datasets --load-alignments \
    --share-all-embeddings --max-source-positions 1024 --max-target-positions 1024 \
    --criterion "label_smoothed_cross_entropy" --label-smoothing 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr-scheduler inverse_sqrt --lr ${LR} --warmup-init-lr 1e-07 --stop-min-lr 1e-09 \
    --warmup-updates 4000 --max-update ${MAX_UPDATES} --max-epoch ${MAX_EPOCH} \
    --attention-dropout 0.1 --dropout 0.1 --max-tokens ${MAX_TOKENS} --update-freq ${UPDATE_FREQ} \
    --seed 1 --log-format simple --skip-invalid-size-inputs-valid-test --fp16 \
    --bpe sentencepiece --sentencepiece-model ${SPM_MODEL} --subword-prob ${SUBWORD_PROB} \
    --max-slot-num ${MAX_SLOT_NUM} --slot-prob ${SLOT_PROB} --slot-method ${SLOT_METHOD} \
    --ddp-backend=no_c10d 2>&1 | tee -a ${SAVE_DIR}/train.log
fi

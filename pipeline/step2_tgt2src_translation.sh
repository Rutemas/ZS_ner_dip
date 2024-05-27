export CUDA_VISIBLE_DEVICES=0

DATA_BIN="path/to/langs1/"
SPM_MODEL="path/to/pretrained_model/flores/sentencepiece.bpe.model"
INPUT_DIR="path/to/xlmr/translation/X/"
OUTPUT_DIR="path/to/xlmr/translation/BT/"
mkdir -p ${OUTPUT_DIR}

src="ru"
tgt="en"
beam=1
nbest=1
MODEL="path/to/pretrained_model/flores/model.pt"
INPUT_FILE="ru.txt0000"
OUTPUT_FILE="ru0000.2en"
BATCH_SIZE=256

INPUT="${INPUT_DIR}/${INPUT_FILE}"
OUTPUT="${OUTPUT_DIR}/${OUTPUT_FILE}"
BUFFER_SIZE=10000
lenpen=1.0

echo "${src}->${tgt} | beam $beam | MODEL ${MODEL}"
echo "INPUT ${INPUT} | OUTPUT ${OUTPUT}"
echo "BATCH_SIZE ${BATCH_SIZE} | BUFFER_SIZE ${BUFFER_SIZE}"

LANGS="en,ru"
LANG_PAIRS="en-ru,ru-en"

cat $INPUT | python3.7 ./m2m/fairseq_cli/interactive.py ${DATA_BIN} \
  --path ${MODEL} \
  --encoder-langtok "src" --langtoks '{"main":("src", "tgt")}' \
  --task "translation_multi_simple_epoch" \
  --langs $LANGS --truncate-source \
  --lang-pairs ${LANG_PAIRS} --max-len-b 256 --min-len 2 --nbest ${nbest} \
  --source-lang ${src} --target-lang ${tgt} \
  --buffer-size ${BUFFER_SIZE} --batch-size ${BATCH_SIZE} \
  --beam ${beam} --lenpen ${lenpen} --unkpen 10000 \
  --no-progress-bar --fp16 --bpe sentencepiece --sentencepiece-model ${SPM_MODEL} > ${OUTPUT}.log

cat ${OUTPUT}.log | grep -P "^D" | cut -f 3- > ${OUTPUT}
echo "Successfully saving ${INPUT} to ${OUTPUT}..."

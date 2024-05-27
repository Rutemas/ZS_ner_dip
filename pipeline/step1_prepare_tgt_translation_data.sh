LANGS=("ru")

GENERATE_SCRIPT="m2m/scripts/pattern/m2m/ner/prepare_insert_pattern_data.py"
INPUT_DIR="path/to/xlmr/data-bin/"
OUTPUT_DIR="path/to/xlmr/translation/X/"
mkdir -p ${OUTPUT_DIR}

for lg in ${LANGS[@]}; do
  INPUT="${INPUT_DIR}/train-${lg}.tsv"
  OUTPUT="${OUTPUT_DIR}/${lg}.txt0000"

  echo "Converting ${INPUT} -> ${OUTPUT}"
  python3.7 ${GENERATE_SCRIPT} -input ${INPUT} -raw-sentence ${OUTPUT} -output "" -entity ""\
  -sentencepiece-model "path/to/pretrained_model/flores/sentencepiece.bpe.model"
done
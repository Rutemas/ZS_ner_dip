LANGS=("ru")

GENERATE_SCRIPT="m2m/scripts/pattern/m2m/ner/prepare_insert_pattern_data.py"
SPM_MODEL="path/to/pretrained_model/flores/sentencepiece.bpe.model"
INPUT_DIR="path/to/xlmr/translation/NER/"
OUTPUT_DIR="path/to/xlmr/translation/LABELED_EN/"
mkdir -p ${OUTPUT_DIR}

INPUT="${INPUT_DIR}/en/test.xlmr.tsv"
OUTPUT="${OUTPUT_DIR}/en.txt0000"

echo "${INPUT} -> ${OUTPUT} + ${ENTITY}"
python3.7 ${GENERATE_SCRIPT} -input ${INPUT} -output ${OUTPUT} \
  -entity "" -raw-sentences "" -lang "en" -sentencepiece-model ${SPM_MODEL}

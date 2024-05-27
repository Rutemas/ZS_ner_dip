GENERATE_SCRIPT="m2m/scripts/pattern/m2m/ner/prepare_ner_data.py"
SPM_MODEL="path/to/pretrained_model/flores/sentencepiece.bpe.model"
INPUT_DIR="path/to/xlmr/translation/BT"
OUTPUT_DIR="path/to/xlmr/translation/NER"
mkdir -p ${OUTPUT_DIR}

INPUT="${INPUT_DIR}/ru0000.2en"
OUTPUT="${OUTPUT_DIR}/en/test.xlmr"
IDX="$OUTPUT_DIR/en/test.xlmr.idx"

echo "${INPUT} -> ${OUTPUT} + ${IDX}"
python3.7 ${GENERATE_SCRIPT} -input ${INPUT} -output ${OUTPUT} -idx ${IDX}

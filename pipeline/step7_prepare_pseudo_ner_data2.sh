LANGS=("ru")

GENERATE_SCRIPRT="m2m/scripts/pattern/m2m/ner/generate_ner_idx.py"
INPUT_DIR="path/to/xlmr/"
OUTPUT_DIR="path/to/xlmr/translation/X_NER/"

for lg in ${LANGS[@]}; do
  INPUT="${INPUT_DIR}/train-${lg}.tsv"
  OUTPUT="${OUTPUT_DIR}/${lg}/test.xlmr"
  mkdir -p ${OUTPUT_DIR}/${lg}/

  echo "Copying ${INPUT} -> ${OUTPUT}"
  cp ${INPUT} ${OUTPUT}

  echo "Generating IDX FLLE"
  python3.7 ${GENERATE_SCRIPRT} -input ${OUTPUT_DIR}/${lg}/test.xlmr
done

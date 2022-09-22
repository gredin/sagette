SOURCE_FILE="officiel-du-scrabble-8.txt"
OUTPUT_FILE="dev_lexicon.txt"

rm ${OUTPUT_FILE}
cat "${SOURCE_FILE}" | sed -nr '/^.{2}$/p' | shuf -n 100 >> ${OUTPUT_FILE}
cat "${SOURCE_FILE}" | sed -nr '/^.{3}$/p' | shuf -n 200 >> ${OUTPUT_FILE}
cat "${SOURCE_FILE}" | sed -nr '/^.{4}$/p' | shuf -n 300 >> ${OUTPUT_FILE}
cat "${SOURCE_FILE}" | sed -nr '/^.{5}$/p' | shuf -n 400 >> ${OUTPUT_FILE}
cat "${SOURCE_FILE}" | sed -nr '/^.{6}$/p' | shuf -n 500 >> ${OUTPUT_FILE}
cat "${SOURCE_FILE}" | sed -nr '/^.{7}$/p' | shuf -n 600 >> ${OUTPUT_FILE}
cat "${SOURCE_FILE}" | sed -nr '/^.{8}$/p' | shuf -n 700 >> ${OUTPUT_FILE}
cat "${SOURCE_FILE}" | sed -nr '/^.{9}$/p' | shuf -n 800 >> ${OUTPUT_FILE}
cat "${SOURCE_FILE}" | sed -nr '/^.{10}$/p' | shuf -n 900 >> ${OUTPUT_FILE}

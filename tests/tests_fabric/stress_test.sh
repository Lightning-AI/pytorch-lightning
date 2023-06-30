test_batch_size="${PL_STANDALONE_TESTS_BATCH_SIZE:-6}"
test_iters="${ITERS:-10}"

export PL_RUN_STANDALONE_TESTS=1

parametrizations=$(python -m pytest --collect-only --quiet "$@")
if [[ ! $parametrizations == *"1 test collected"* ]]; then
    echo "Failed to isolate a single test:"
    echo $parametrizations
    exit 1
fi

for (( iter=1; iter<$test_iters; iter++)); do
    for (( i=1; i<=$test_batch_size; i++ )); do
        OUTPUT_FILE="standalone_test_output_${i}.txt"
        rm -f ${OUTPUT_FILE}
        python -m pytest -vvv -s "$@" &>> ${OUTPUT_FILE} &
        pids[${i}]=$!
    done

    for (( i=1; i<=$test_batch_size; i++ )); do
        wait ${pids[$i]}
        retcodes[${i}]=$?
    done

    success=1
    for (( i=1; i<=$test_batch_size; i++ )); do
        if [ ${retcodes[$i]} -ne 0 ]; then
            echo "Failed: standalone_test_output_${i}.txt"
            success=0
        fi
    done
    if [ $success -ne 1 ]; then
        exit 1
    fi

    unset pids  # empty the array
    unset retcodes
    echo "Iter ${iter} complete"
done

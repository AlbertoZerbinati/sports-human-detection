#!/bin/bash

# Alberto Zerbinati

# Function to create directories if they don't exist
make_dir() {
    if [ ! -d "$1" ]; then
        mkdir -p "$1"
    fi
}

# Source and target directories
SOURCE_DIR="data"
TARGET_DIR="data/dataset"

# Create target directories
make_dir "${TARGET_DIR}/train/negative"
make_dir "${TARGET_DIR}/train/positive"
make_dir "${TARGET_DIR}/valid/negative"
make_dir "${TARGET_DIR}/valid/positive"
make_dir "${TARGET_DIR}/test/negative"
make_dir "${TARGET_DIR}/test/positive"

# Split ratios
TRAIN_RATIO=0.85
VALID_RATIO=0.14
TEST_RATIO=0.01

# Function to split files
split_files() {
    local SOURCE=$1
    local TARGET=$2
    local TOTAL_FILES=$(ls $SOURCE | wc -l)
    local TRAIN_COUNT=$(awk "BEGIN {print int(${TOTAL_FILES} * ${TRAIN_RATIO})}")
    local VALID_COUNT=$(awk "BEGIN {print int(${TOTAL_FILES} * ${VALID_RATIO})}")
    local TEST_COUNT=$(awk "BEGIN {print int(${TOTAL_FILES} * ${TEST_RATIO})}")
    
    # Shuffle and copy files
    shuf -e ${SOURCE}/* | head -n ${TRAIN_COUNT} | xargs -I {} cp {} ${TARGET}/train/$3
    shuf -e ${SOURCE}/* | head -n ${VALID_COUNT} | xargs -I {} cp {} ${TARGET}/valid/$3
    shuf -e ${SOURCE}/* | tail -n ${TEST_COUNT} | xargs -I {} cp {} ${TARGET}/test/$3
}

# Split negative and positive files
split_files "${SOURCE_DIR}/negative" "${TARGET_DIR}" "negative"
split_files "${SOURCE_DIR}/positive" "${TARGET_DIR}" "positive"


# Function to count files in a directory
count_files() {
    ls $1 | wc -l
}

# Count and output the number of files in each split
TRAIN_POSITIVE=$(count_files "${TARGET_DIR}/train/positive")
TRAIN_NEGATIVE=$(count_files "${TARGET_DIR}/train/negative")
TRAIN_TOTAL=$((TRAIN_POSITIVE + TRAIN_NEGATIVE))

VALID_POSITIVE=$(count_files "${TARGET_DIR}/valid/positive")
VALID_NEGATIVE=$(count_files "${TARGET_DIR}/valid/negative")
VALID_TOTAL=$((VALID_POSITIVE + VALID_NEGATIVE))

TEST_POSITIVE=$(count_files "${TARGET_DIR}/test/positive")
TEST_NEGATIVE=$(count_files "${TARGET_DIR}/test/negative")
TEST_TOTAL=$((TEST_POSITIVE + TEST_NEGATIVE))

echo "done splitting dataset! 🎉"
echo "train: ${TRAIN_POSITIVE} positive, ${TRAIN_NEGATIVE} negative ($((TRAIN_TOTAL * 100 / (TRAIN_TOTAL + VALID_TOTAL + TEST_TOTAL)))%)"
echo "valid: ${VALID_POSITIVE} positive, ${VALID_NEGATIVE} negative ($((VALID_TOTAL * 100 / (TRAIN_TOTAL + VALID_TOTAL + TEST_TOTAL)))%)"
echo "test: ${TEST_POSITIVE} positive, ${TEST_NEGATIVE} negative ($((TEST_TOTAL * 100 / (TRAIN_TOTAL + VALID_TOTAL + TEST_TOTAL)))%)"

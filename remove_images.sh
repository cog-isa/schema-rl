VIS_DIR="./visualization"

TARGET_DIRS="
attribute_schemas/
backtracking/
backtracking_schemas/
entities/
logs/
replay_buffer/
reward_schemas/
state/
"

for DIR in $TARGET_DIRS
do
    find "$VIS_DIR/$DIR" -name "*" -type f -delete
done

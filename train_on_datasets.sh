jq -r '.[]' datasets/dataset_names.json | while read -r dataset_name; do
    echo "Training on $dataset_name"
    python train.py "$dataset_name" test_run --loader forecast_npy --gpu -1 || exit 1
done
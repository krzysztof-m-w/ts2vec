n_passes=100

for pass in $(seq 1 $n_passes); do
    i=0
    echo "Pass $pass/$n_passes"
    jq -r '.[]' datasets/dataset_names.json | while read -r leave_dataset_name; do
        jq -r '.[]' datasets/dataset_names.json | while read -r dataset_name; do

            if [[ "$leave_dataset_name" != "$dataset_name" ]]; then
                echo "Training on $dataset_name $i/17 $pass/$n_passes"
                python train.py "$dataset_name" test_run --loader forecast_npy --gpu -1 --load-model "runs/run_${leave_dataset_name}_left" --epochs 1 || exit 1
            fi

        done

        ((i++))
    done
done
graph=${1:-Biseasome}
lr=${2:-0.01}
bs=${3:-1}
weight_decay=${4:-0}

models="euclidean poincare prod-hyhy prod-hyeu spd"
#models="spd"

for model in $models
do
  for th in $tree_heights
  do
    if [[ "$model" =~ ^(spd)$ ]]; then
        dims=8
    else
        dims=36
    fi
      
    if [[ "$model" =~ ^(euclidean|prod-hyeu)$ ]]; then
        metrics="l1 l2 linf"
    else
        metrics="l1"
    fi

    for metric in $metrics
    do
        echo $model $metric $dims $graph $lr $weight_decay $th $bs 
        python run_dis.py \
            --dims $dims \
            --graph $graph \
            --model $model \
            --metric $metric \
            --learning_rate $lr \
            --weight_decay $weight_decay \
            --batch_size $bs \
            --epoch 2000
    done   
  done
done
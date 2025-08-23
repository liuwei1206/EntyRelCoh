# <<"COMMENT"
python3 train.py --do_train \
                         --dataset="toefl_p1" \
                         --fold_id=1 \
                         --label_list="low, medium, high" \
                         --model_type="base" \
                         --model_name_or_path="roberta-base" \
                         --max_text_length=512 \
                         --max_sent_num=24 \
                         --max_rel_num=56 \
                         --num_train_epochs=20 \
                         --train_batch_size=32 \
                         --learning_rate=1e-3 \
                         --dropout=0.2

# COMMENT

# for roberta-base
<<"COMMENT"
for fold in 1 2 3 4 5
do
    python3 train.py --do_train \
                             --dataset="toefl_p1" \
                             --fold_id=${fold} \
                             --label_list="low, medium, high" \
                             --model_type="base" \
                             --model_name_or_path="roberta-base" \
                             --max_sent_num=24 \
                             --max_rel_num=56 \
                             --num_train_epochs=20 \
                             --train_batch_size=32 \
                             --learning_rate=1e-3 \
                             --dropout=0.2
done
COMMENT


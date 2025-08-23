<<"COMMENT"
python3 train_entyrel.py --do_train \
                         --dataset="enron" \
                         --fold_id=2 \
                         --label_list="1, 2, 3" \
                         --model_type="fusion" \
                         --no_rel \
                         --model_name_or_path="Llama-3.1-8B-Instruct" \
			                   --max_text_length=768 \
                         --max_sent_num=15 \
                         --max_rel_num=32 \
                         --num_train_epochs=20 \
                         --train_batch_size=32 \
                         --learning_rate=1e-3 \
			                   --dropout=0.1

COMMENT

# for roberta-base
<<"COMMENT"
for fold in 1 2 3 4 5 6 7 8 9 10
do
    python3 train_entyrel.py --do_train \
                             --dataset="enron" \
                             --fold_id=${fold} \
                             --label_list="1, 2, 3" \
                             --model_type="fusion" \
			                       --no_rel \
                             --model_name_or_path="Llama-3.1-8B-Instruct" \
                             --max_text_length=768 \
                             --max_sent_num=15 \
                             --max_rel_num=32 \
                             --num_train_epochs=20 \
                             --train_batch_size=32 \
                             --learning_rate=1e-3 \
                             --dropout=0.1
done
COMMENT


# <<"COMMENT"
for task in clinton yahoo yelp
do
    python3 train_entyrel.py --do_test \
                             --dataset=${task} \
                             --fold_id=9 \
                             --label_list="1, 2, 3" \
                             --model_type="fusion" \
			                       --no_rel \
                             --model_name_or_path="Llama-3.1-8B-Instruct" \
                             --max_text_length=768 \
                             --max_sent_num=15 \
                             --max_rel_num=32 \
                             --num_train_epochs=20 \
                             --train_batch_size=32 \
                             --learning_rate=1e-3 \
                             --dropout=0.1
done
# COMMENT

<<"COMMENT"
python3 train_entyrel.py --do_train \
                         --dataset="cohe" \
                         --fold_id=1 \
                         --label_list="low, medium, high" \
                         --model_type="fusion" \
                         --no_rel \
                         --model_name_or_path="Llama-3.1-8B-Instruct" \
			                   --max_text_length=768 \
                         --max_sent_num=15 \
                         --max_rel_num=36 \
                         --num_train_epochs=20 \
                         --train_batch_size=32 \
                         --learning_rate=1e-3 \
			                   --dropout=0.1

COMMENT

# for roberta-base
# <<"COMMENT"
for fold in 1 2 3 4 5
do
    python3 train_entyrel.py --do_train \
                             --dataset="cohe" \
                             --fold_id=${fold} \
                             --label_list="low, medium, high" \
                             --model_type="fusion" \
			                       --no_rel \
                             --model_name_or_path="Llama-3.1-8B-Instruct" \
                             --max_text_length=768 \
                             --max_sent_num=15 \
                             --max_rel_num=36 \
                             --num_train_epochs=20 \
                             --train_batch_size=32 \
                             --learning_rate=1e-3 \
                             --dropout=0.1
done
# COMMENT


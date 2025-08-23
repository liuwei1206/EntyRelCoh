<<"COMMENT"
python3 train_entyrel.py --do_train \
                         --dataset="toefl_p5" \
                         --fold_id=2 \
                         --label_list="low, medium, high" \
                         --model_type="fusion" \
                         --no_enty \
                         --model_name_or_path="Llama-3.1-8B-Instruct" \
			                   --max_text_length=1024 \
                         --max_sent_num=24 \
                         --max_rel_num=56 \
                         --num_train_epochs=20 \
                         --train_batch_size=32 \
                         --learning_rate=1e-3 \
			                   --dropout=0.1

COMMENT

# for roberta-base
<<"COMMENT"
for task in p1 p2 p3
do
for fold in 1 2 3 4 5
do
    python3 train_entyrel.py --do_train \
                             --dataset="toefl_${task}" \
                             --fold_id=${fold} \
                             --label_list="low, medium, high" \
                             --model_type="fusion" \
			                       --no_enty \
                             --model_name_or_path="Llama-3.1-8B-Instruct" \
			                       --max_text_length=1024 \
                             --max_sent_num=24 \
                             --max_rel_num=56 \
                             --num_train_epochs=20 \
                             --train_batch_size=32 \
                             --learning_rate=1e-3 \
                             --dropout=0.1
done
done
COMMENT

# <<"COMMENT"
for task in p1 p2 p3 p4 p6 p7 p8
do
    python3 train_entyrel.py --do_test \
                             --dataset="toefl_${task}" \
                             --fold_id=2 \
                             --label_list="low, medium, high" \
                             --model_type="fusion" \
			                       --no_enty \
                             --model_name_or_path="Llama-3.1-8B-Instruct" \
                             --max_text_length=1024 \
                             --max_sent_num=24 \
                             --max_rel_num=56 \
                             --num_train_epochs=20 \
                             --train_batch_size=32 \
                             --learning_rate=1e-3 \
                             --dropout=0.2
done
# COMMENT

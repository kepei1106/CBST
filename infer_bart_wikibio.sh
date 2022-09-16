GPU=0

model_path='bart_few50'
for run in `seq 0 1 3`
do
  CUDA_VISIBLE_DEVICES=${GPU} python cli_gt.py \
    --do_predict \
    --output_dir ./out/wikibio/${model_path}/model_label_${run} \
    --train_file ./data/wikibio/train_50 \
    --predict_file ./data/wikibio/test \
    --model_path ./bart_model \
    --tokenizer_path ./bart_model \
    --predict_batch_size 32 \
    --max_input_length 384 \
    --max_output_length 128 \
    --append_another_bos \
    --num_beams 1 \
    --prefix test_beam1_lenpen1_
done


model_path='bart_few100'
for run in `seq 0 1 3`
do
  CUDA_VISIBLE_DEVICES=${GPU} python cli_gt.py \
    --do_predict \
    --output_dir ./out/wikibio/${model_path}/model_label_${run} \
    --train_file ./data/wikibio/train_100 \
    --predict_file ./data/wikibio/test \
    --model_path ./bart_model \
    --tokenizer_path ./bart_model \
    --predict_batch_size 32 \
    --max_input_length 384 \
    --max_output_length 128 \
    --append_another_bos \
    --num_beams 1 \
    --prefix test_beam1_lenpen1_
done

model_path='bart_few200'
for run in `seq 0 1 3`
do
  CUDA_VISIBLE_DEVICES=${GPU} python cli_gt.py \
    --do_predict \
    --output_dir ./out/wikibio/${model_path}/model_label_${run} \
    --train_file ./data/wikibio/train_200 \
    --predict_file ./data/wikibio/test \
    --model_path ./bart_model \
    --tokenizer_path ./bart_model \
    --predict_batch_size 32 \
    --max_input_length 384 \
    --max_output_length 128 \
    --append_another_bos \
    --num_beams 1 \
    --prefix test_beam1_lenpen1_ 
done

model_path='bart_few500'
for run in `seq 0 1 3`
do
  CUDA_VISIBLE_DEVICES=${GPU} python cli_gt.py \
    --do_predict \
    --output_dir ./out/wikibio/${model_path}/model_label_${run} \
    --train_file ./data/wikibio/train_500 \
    --predict_file ./data/wikibio/test \
    --model_path ./bart_model \
    --tokenizer_path ./bart_model \
    --predict_batch_size 32 \
    --max_input_length 384 \
    --max_output_length 128 \
    --append_another_bos \
    --num_beams 1 \
    --prefix test_beam1_lenpen1_ 
done

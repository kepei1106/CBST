GPU=0

model_path='bart_few001'
for run in `seq 0 1 3`
do
  CUDA_VISIBLE_DEVICES=$GPU python cli_gt.py \
        --do_predict \
        --output_dir ./out/webnlg/$model_path/model_label_$run \
        --train_file ./data/webnlg/train_001 \
        --predict_file ./data/webnlg/test \
        --model_path ./bart_model \
        --tokenizer_path ./bart_model \
        --predict_batch_size 32 \
        --max_input_length 256 \
        --max_output_length 128 \
        --append_another_bos \
        --num_beams 5 \
        --prefix test_beam5_lenpen1_ 

done

model_path='bart_few0005'
for run in `seq 0 1 3`
do
  CUDA_VISIBLE_DEVICES=$GPU python cli_gt.py \
        --do_predict \
        --output_dir ./out/webnlg/$model_path/model_label_$run \
        --train_file ./data/webnlg/train_0005 \
        --predict_file ./data/webnlg/test \
        --model_path ./bart_model \
        --tokenizer_path ./bart_model \
        --predict_batch_size 32 \
        --max_input_length 256 \
        --max_output_length 128 \
        --append_another_bos \
        --num_beams 5 \
        --prefix test_beam5_lenpen1_ 

done

model_path='bart_few005'
for run in `seq 0 1 3`
do
  CUDA_VISIBLE_DEVICES=$GPU python cli_gt.py \
        --do_predict \
        --output_dir ./out/webnlg/$model_path/model_label_$run \
        --train_file ./data/webnlg/train_005 \
        --predict_file ./data/webnlg/test \
        --model_path ./bart_model \
        --tokenizer_path ./bart_model \
        --predict_batch_size 32 \
        --max_input_length 256 \
        --max_output_length 128 \
        --append_another_bos \
        --num_beams 5 \
        --prefix test_beam5_lenpen1_ 

done

model_path='bart_few01'
for run in `seq 0 1 3`
do
  CUDA_VISIBLE_DEVICES=$GPU python cli_gt.py \
        --do_predict \
        --output_dir ./out/webnlg/$model_path/model_label_$run \
        --train_file ./data/webnlg/train_01 \
        --predict_file ./data/webnlg/test \
        --model_path ./bart_model \
        --tokenizer_path ./bart_model \
        --predict_batch_size 32 \
        --max_input_length 256 \
        --max_output_length 128 \
        --append_another_bos \
        --num_beams 5 \
        --prefix test_beam5_lenpen1_ 

done

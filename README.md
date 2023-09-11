## Environment
- `python 3.8`
- `pip install -r requirements.txt`

## Train Model
Prompt: [CLS] prompt [MASK] doc [SEP]
```
python main.py --model_name LongFormerPrompt \
               --gpu 0 \
               --bert_path IDEA-CCNL/Erlangshen-Longformer-110M \
               --test_freq 10 \
               --data_path data/ChineseEMR-50 \
               --embedding_dim 768 \
               --data_loader DataLoader_prompt_front \
               --max_length 1020 \
               --dropout_rate 0.5 \
               --epochs 40 \
               --batch_size 12 \
               --bert_lr 1e-5 \
               --other_lr 5e-4 \
               --warmup_rate 0.3 \
               --patience 5 \
               --seed 2 \
               --use_wandb \
               --data_version 2 \
               --result_path result \
               --accumulation_steps 1 \
               --label_smooth_lambda 0.0 \
               --hidden_size 768
```
### Trained model

The model weight file trained using the above instruction: https://drive.google.com/file/d/12oAtq6Cu5fEPtKxH7X6Y_ytGO4OKY9u0/view?usp=drive_link

## Sample input
ChineseEMR-50:`data/ChineseEMR-50/tiny.json`

Small-ChineseEMR:  `data/Small-ChineseEMR/tiny.json`


load_model='/home/zebraclips/yynil/models/rwkv7-g1a-1.5b-20250922-ctx4096.pth'
proj_dir='/home/zebraclips/yynil/demo/states_tuning'
data_file='/home/zebraclips/yynil/github/RWKV-PEFT/demo_data/translation/'
tokenizer_dir='/home/zebraclips/yynil/models/rwkv7-1.5B-g1a'


n_layer=24
n_embd=2048

micro_bsz=2
epoch_save=1
epoch_steps=800
ctx_len=512


python train.py --load_model $load_model \
--proj_dir $proj_dir --data_file $data_file \
--vocab_size 65536 \
--n_layer $n_layer --n_embd $n_embd \
--data_type jsonl_dir --dataload pad --loss_mask pad \
--ctx_len $ctx_len --micro_bsz $micro_bsz \
--epoch_steps $epoch_steps --epoch_count 10 --epoch_begin 0 --epoch_save $epoch_save \
--lr_init 1 --lr_final 1e-2 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
--accelerator gpu --devices 1 --precision bf16 --strategy deepspeed_stage_2_offload --grad_cp 1 \
--my_testing "x070" \
--train_type "state"  --dataload pad --op fla --tokenizer_dir $tokenizer_dir 
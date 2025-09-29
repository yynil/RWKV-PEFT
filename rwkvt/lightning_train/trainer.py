import os, math, time, datetime, subprocess
import torch
from torch.utils.data import DataLoader
from lightning_utilities.core.rank_zero import rank_zero_info, rank_zero_only
import lightning as pl
import json
from rwkvt.trick.lrs import wsd,cos_decay

def my_save(args, trainer, dd, ff):
    if '14b-run1' in ff:
        fn = ff.split('/')[-1]
        fff = '/dev/shm/' + fn
        torch.save(dd, fff)
        subprocess.Popen(f" aws s3 mv {fff} s3://rwkv-14b-4k/{fn} --quiet", shell=True)
    elif ('world/14b' in ff) or ('world/7b' in ff):
        aa = ff.split('/')[1]
        fn = ff.split('/')[-1]
        fff = f'/dev/shm/{aa}-{fn}'
        torch.save(dd, fff)
        subprocess.Popen(f" aws s3 mv {fff} s3://rwkv-world/{aa}-{fn} --quiet", shell=True)
    else:
        torch.save(dd, ff)


def save_ckpt(args, trainer, pl_module, step_suffix=""):
    """保存检查点函数
    
    Args:
        args: 训练参数
        trainer: PyTorch Lightning trainer
        pl_module: PyTorch Lightning module
        step_suffix: 可选的步骤后缀，用于在epoch中间保存时区分
    """
    to_save_dict = {}
    
    if args.data_type == 'wds_img':
        raw_dict = pl_module.state_dict()
        for k in raw_dict:
            if k.startswith('encoder.') or k.startswith('decoder.'):
                to_save_dict[k] = raw_dict[k]
    else:
        # to_save_dict = pl_module.state_dict()
        to_save_dict = {k.replace("model.", ""): v for k, v in pl_module.state_dict().items()}

    if args.train_type=='state':
        peft_dict = {}
        for name, state in to_save_dict.items():
            if 'state' in name:
                peft_dict[name] = state
        to_save_dict = peft_dict

    if args.peft!='none':
        peft_dict = {}
        for name, state in to_save_dict.items():
            if len(args.load_model) == 0:
                if 'emb' in name or 'head' in name or 'ln' in name:
                    peft_dict[name] = state
            for part in args.train_parts:
                if part in name:
                    peft_dict[name] = state
            if args.peft=='pissa' and ('lora' in name):
                peft_dict[name] = state
            elif args.peft in name:
                peft_dict[name] = state

        to_save_dict = peft_dict

    # 构建文件名，包含epoch和steps信息
    epoch = args.epoch_begin + trainer.current_epoch
    steps = trainer.global_step
    filename = f"rwkv-{epoch}-step{steps}"
    if step_suffix:
        filename += f"-{step_suffix}"
    filename += ".pth"

    try:
        my_save(
            args, trainer,
            to_save_dict,
            f"{args.proj_dir}/{filename}",
        )
    except Exception as e:
        print('Error\n\n', e, '\n\n')
        


class train_callback(pl.Callback):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.loss_file = os.path.join(args.proj_dir, "loss_data.jsonl")
        if os.path.exists(self.loss_file):
            os.remove(self.loss_file)
            
    def write_data(self, loss_data, t_cost, kt_s):
        # 将loss数据写入文件，便于streamlit绘图
        with open(self.loss_file, 'a') as f:
            json.dump({"loss": float(loss_data), "t_cost": t_cost, "kt_s": kt_s}, f)
            f.write('\n')

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        args = self.args
        # if args.cuda_cleanup > 0:
        #     torch.cuda.empty_cache()
        real_step = trainer.global_step + args.epoch_begin * args.epoch_steps

        # LR schedule
        w_step = args.warmup_steps
        if args.lr_final == args.lr_init or args.epoch_count == 0:
            lr = args.lr_init
        else:
            if 'wsd' == args.lr_schedule:
                lr = wsd(args.lr_init, 0, real_step, (args.epoch_steps*args.epoch_count)//int(args.devices)//args.accumulate_grad_batches,warmup_steps=w_step)
            else:
                lr = cos_decay(args.lr_init, args.lr_final, real_step, (args.epoch_steps*args.epoch_count)//int(args.devices)//args.accumulate_grad_batches)



        if args.weight_decay_final > 0:
            wd_now = args.weight_decay * math.exp(math.log(args.weight_decay_final / args.weight_decay) * progress)
        else:
            wd_now = args.weight_decay

        for param_group in trainer.optimizers[0].param_groups:
            if param_group["weight_decay"] > 0:
                param_group["weight_decay"] = wd_now
            if args.layerwise_lr > 0:
                param_group["lr"] = lr * param_group["my_lr_scale"]
                # print(param_group["lr"], param_group["my_lr_scale"])
            else:
                param_group["lr"] = lr

        trainer.my_lr = lr
        trainer.my_wd = wd_now
        # rank_zero_info(f"{real_step} {lr}")

        if trainer.global_step == 0:
            if trainer.is_global_zero:  # logging
                trainer.my_loss_sum = 0
                trainer.my_loss_count = 0
                trainer.my_log = open(args.proj_dir + "/train_log.txt", "a")
                trainer.my_log.write(f"NEW RUN {args.my_timestamp}\n{vars(self.args)}\n")
                try:
                    print(f"\n{trainer.strategy.config}\n")
                    trainer.my_log.write(f"{trainer.strategy.config}\n")
                except:
                    pass
                trainer.my_log.flush()
                if len(args.wandb) > 0:
                    print("Login to wandb...")
                    import wandb
                    wandb.init(
                        project=args.wandb,
                        name=args.run_name + " " + args.my_timestamp,
                        config=args,
                        save_code=False,
                    )
                    trainer.my_wandb = wandb

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        args = self.args
        token_per_step = pl_module.tokens_trained_this_step * args.devices
        real_step = trainer.global_step + args.epoch_begin * args.epoch_steps

        if pl.__version__[0]=='2' :
            loss = outputs['loss']
            if int(args.devices)>1:
                torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)

        if trainer.is_global_zero:  # logging
            t_now = time.time_ns()
            kt_s = 0
            t_cost = 0
            try:
                t_cost = (t_now - trainer.my_time_ns) / 1e9
                kt_s = token_per_step / t_cost / 1000
                t_cost = 1.0 / t_cost
                self.log("REAL it/s", t_cost, prog_bar=True, on_step=True)
                self.log("Kt/s", kt_s, prog_bar=True, on_step=True)
            except:
                pass
            trainer.my_time_ns = t_now
            if pl.__version__[0]=='2':
                trainer.my_loss = loss*trainer.accumulate_grad_batches/int(args.devices)
            else:
                trainer.my_loss = trainer.my_loss_all.float().mean().item()
            trainer.my_loss_sum += trainer.my_loss
            trainer.my_loss_count += 1
            trainer.my_epoch_loss = trainer.my_loss_sum / trainer.my_loss_count
            self.log("lr", trainer.my_lr, prog_bar=True, on_step=True)
            self.log("sum_loss", trainer.my_epoch_loss, prog_bar=True, on_step=True)
            self.log("loss", trainer.my_loss, prog_bar=True, on_step=True)

            # 将loss、t_cost、kt_s写入data.json
            if trainer.accumulate_grad_batches!=None:
                args.avg_loss += trainer.my_loss / trainer.accumulate_grad_batches
                if (batch_idx+1) % trainer.accumulate_grad_batches == 0:
                    if len(args.wandb) > 0:
                        lll = {"loss": args.avg_loss, "lr": trainer.my_lr, "wd": trainer.my_wd, "Gtokens": real_step * token_per_step / 1e9}
                        if kt_s > 0:
                            lll["kt/s"] = kt_s
                        trainer.my_wandb.log(lll, step=int(real_step))
                    self.write_data(args.avg_loss, t_cost, kt_s)
                    args.avg_loss = 0
            else:
                if len(args.wandb) > 0:
                    lll = {"loss": trainer.my_loss, "lr": trainer.my_lr, "wd": trainer.my_wd, "Gtokens": real_step * token_per_step / 1e9}
                    if kt_s > 0:
                        lll["kt/s"] = kt_s
                    trainer.my_wandb.log(lll, step=int(real_step))
                self.write_data(trainer.my_loss, t_cost, kt_s)
            
            # 根据save_per_steps进行中间保存
            if hasattr(args, 'save_per_steps') and args.save_per_steps > 0:
                if trainer.global_step > 0 and trainer.global_step % args.save_per_steps == 0:
                    if (trainer.is_global_zero) or ('deepspeed_stage_3' in args.strategy):
                        save_ckpt(args, trainer, pl_module, "step")
                

    def on_train_epoch_start(self, trainer, pl_module):
        args = self.args
        if args.data_type == "jsonl_dir":
            return
        else:
            if pl.__version__[0]=='2':
                dataset = trainer.train_dataloader.dataset
            else:
                dataset = trainer.train_dataloader.dataset.datasets
            assert "MyDataset" in str(dataset)
            dataset.global_rank = trainer.global_rank
            dataset.real_epoch = int(args.epoch_begin + trainer.current_epoch)
            dataset.world_size = trainer.world_size
            # print(f'########## world_size {dataset.world_size} global_rank {dataset.global_rank} real_epoch {dataset.real_epoch} ##########')

    def on_train_epoch_end(self, trainer, pl_module):
        args = self.args
        if (trainer.is_global_zero) or ('deepspeed_stage_3' in args.strategy):  # save pth
            if (args.epoch_save > 0 and trainer.current_epoch % args.epoch_save == 0) or (trainer.current_epoch == args.epoch_count - 1):
                save_ckpt(args, trainer, pl_module)

        if trainer.is_global_zero:  # logging
            trainer.my_log.write(f"{args.epoch_begin + trainer.current_epoch} {trainer.my_epoch_loss:.6f} {math.exp(trainer.my_epoch_loss):.4f} {trainer.my_lr:.8f} {datetime.datetime.now()} {trainer.current_epoch}\n")
            trainer.my_log.flush()

            trainer.my_loss_sum = 0
            trainer.my_loss_count = 0
            if (args.epoch_begin + trainer.current_epoch) >= args.my_exit:
                exit(0)


@rank_zero_only
def generate_init_weight(model, init_weight_name):
    mm = model.generate_init_weight()

    if model.args.my_pile_stage == 1:
        if len(model.args.load_model) > 0:
            print(f"Combine weights from {model.args.load_model}...")
            load_dict = torch.load(model.args.load_model, map_location="cpu")
            for k in load_dict:
                try:
                    assert k in mm
                except:
                    print('missing', k)
                    exit(0)
                src = load_dict[k]
                try:
                    mm[k] = src.reshape(mm[k].shape)
                except:
                    tmp = mm[k].squeeze().clone()
                    print(k, src.shape, '-->', mm[k].shape)
                    ss = src.shape[0]
                    dd = tmp.shape[0]
                    for i in range(dd):
                        pos = i / dd * ss
                        if pos >= ss - 1:
                            tmp[i] = src[ss-1]
                        else:
                            p0 = int(math.floor(pos))
                            ii = pos - p0
                            tmp[i] = src[p0] * (1-ii) + src[p0+1] * (ii)
                    mm[k] = tmp.reshape(mm[k].shape)
                    sss = src.squeeze().float().cpu().numpy()
                    print(sss[:10], '...', sss[-10:])
                    mmm = mm[k].squeeze().float().cpu().numpy()
                    print(mmm[:10], '...', mmm[-10:])

    print(f"Save to {init_weight_name}...")
    torch.save(mm, init_weight_name)

    if model.args.my_pile_stage == 1:
        print("Done. Now go for stage 2.")
        exit(0)


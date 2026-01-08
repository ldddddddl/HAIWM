import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# from joint_states_process import JointStateRead
from model.models import ActNet
import time
from argparse import ArgumentParser
from omegaconf import DictConfig, OmegaConf
import torch.optim as optim
from losses import ComputeLosses
from sklearn.metrics import accuracy_score
from misc import (
    InitAverager,
    set_seed,
    save_images,
    save_checkpoint,
    VariableContainer,
    SegmentBuffer,
    TorchMetricsWrapper,
    MatrixSSIMPSNR,
    update_ssim_psnr,
    tens2act,
    repeat_data,
)
import pandas as pd
import os
from datetime import datetime
from spikingjelly.activation_based import functional
from scipy import stats
import numpy as np
from script.metrics import calc_metrics
from load_jetmax_dataset import load_lerobot_dataloader
import ctypes
from accelerate.utils import send_to_device
from tqdm import tqdm

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. WandB logging will be disabled.")


def setup_ddp():
    """初始化分布式训练环境"""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    """清理分布式训练环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """检查是否为主进程（rank 0）"""
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def get_world_size():
    """获取总进程数"""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def main(xlstm_cfg: DictConfig):
    # ===================== DDP Setup =====================
    use_ddp = getattr(xlstm_cfg, "use_ddp", False)
    local_rank = 0
    if use_ddp:
        local_rank = setup_ddp()
        device = torch.device(f"cuda:{local_rank}")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(xlstm_cfg.cuda_visible_devices)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set seed (ensure reproducibility across ranks)
    seed = set_seed(xlstm_cfg.random_seed)
    xlstm_cfg.random_seed = seed

    is_use_cuda = torch.cuda.is_available()
    model = ActNet(xlstm_cfg, is_use_cuda=is_use_cuda, device=device).to(device=device)

    model.apply(init_weights)
    compute_losses = ComputeLosses(device=device, config=xlstm_cfg).to(device=device)
    optimizer = optim.Adam(
        params=model.parameters(), lr=xlstm_cfg.lr, betas=[0.9, 0.99], weight_decay=1e-4
    )
    if xlstm_cfg.is_resume:
        checkpoint = torch.load(xlstm_cfg.checkpoint_path, map_location=device)
        # Handle DDP wrapped model checkpoint
        state_dict = checkpoint["state_dict"]
        if use_ddp and not any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {f"module.{k}": v for k, v in state_dict.items()}
        elif not use_ddp and any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        model.apply(init_weights)

    # ===================== Wrap model with DDP =====================
    if use_ddp:
        # find_unused_parameters=True is needed because some model parameters
        # may not be used in every forward pass (e.g., when olny_action_generate=True)
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

    writer = ""

    ctime = time.ctime()
    ctime_datetime = datetime.strptime(ctime, "%a %b %d %H:%M:%S %Y")
    formatted_datetime = ctime_datetime.strftime("%y-%m-%d-%H-%M-%S")

    pwd = os.getcwd()

    # Only main process creates directories
    if is_main_process():
        if not os.path.exists(f"{pwd}/{xlstm_cfg.results_path}/{formatted_datetime}"):
            os.makedirs(f"{xlstm_cfg.results_path}/{formatted_datetime}", exist_ok=True)
            os.makedirs(
                f"{xlstm_cfg.results_path}/{formatted_datetime}/check_point",
                exist_ok=True,
            )

    # ===================== WandB Setup (main process only) =====================
    use_wandb = getattr(xlstm_cfg, "use_wandb", False) and WANDB_AVAILABLE
    if is_main_process() and use_wandb:
        wandb_run_name = (
            getattr(xlstm_cfg, "wandb_run_name", None) or formatted_datetime
        )
        wandb.init(
            project=getattr(xlstm_cfg, "wandb_project", "HAIWM"),
            entity=getattr(xlstm_cfg, "wandb_entity", None),
            name=wandb_run_name,
            config=OmegaConf.to_container(xlstm_cfg, resolve=True),
            resume="allow",
        )

    results_dict = {}
    actions_prediction, new_actions_prediction, total_actions = {}, {}, {}
    metrix_results, metrix_frames_result = {}, {}

    # Select dataloader based on dataset type
    use_libero = getattr(xlstm_cfg, "use_libero", False)

    if use_libero:
        # Use LIBERO dataloader for libero datasets
        from load_libero_dataset import load_libero_dataloader

        task_suite = getattr(xlstm_cfg, "task_suite", "libero_10")
        local_dir = xlstm_cfg.datasets_path if xlstm_cfg.datasets_path else None

        dataloader_train, dataloader_valid, normalizer = load_libero_dataloader(
            task_suite=task_suite,
            local_dir=local_dir,
            batch_size=xlstm_cfg.batchsize,
            num_workers=xlstm_cfg.num_workers,
            horizon=xlstm_cfg.horizon,
            past_img_num=xlstm_cfg.past_img_num,
            future_img_num=xlstm_cfg.future_img_num,
            image_size=xlstm_cfg.cropWidth,
            train_ratio=1.0 - xlstm_cfg.valid_datas_scale,
            normalize=True,
            seed=xlstm_cfg.random_seed,
            use_ddp=use_ddp,  # Pass DDP flag to dataloader
        )
        train_ds, valid_ds = None, None  # Not needed for LIBERO
    else:
        # Use LeRobot dataloader for other datasets
        dataloader_train, dataloader_valid, train_ds, valid_ds = (
            load_lerobot_dataloader(
                repo_id=xlstm_cfg.datasets_path,
                local_dir=xlstm_cfg.datasets_path,
                batch_size=xlstm_cfg.batchsize,
                num_workers=xlstm_cfg.num_workers,
                train_ratio=xlstm_cfg.valid_datas_scale,
                seed=xlstm_cfg.random_seed,
                horizon=xlstm_cfg.horizon,
                past_img_num=xlstm_cfg.past_img_num,
                future_img_num=xlstm_cfg.future_img_num,
                normalize=True,
                norm_stats_dir=xlstm_cfg.datasets_path,
                image_size=xlstm_cfg.cropWidth,
                use_ddp=use_ddp,  # Pass DDP flag to dataloader
            )
        )
    grip_upsample_frame_pred_temp, grip_frame_labels_temp = None, None
    side_upsample_frame_pred_temp, side_frame_labels_temp = None, None

    for epc in range(xlstm_cfg.epochs):
        train_avg = InitAverager()
        valid_avg = InitAverager()

        train_phase = ""
        use_tired_training = xlstm_cfg.is_use_three_phase_train
        if use_tired_training:
            if epc / xlstm_cfg.epochs < 0.25 and use_tired_training:
                train_phase = "generate"
            elif 0.25 <= epc / xlstm_cfg.epochs < 0.5 and use_tired_training:
                train_phase = "inference"
            elif 0.5 <= epc / xlstm_cfg.epochs < 0.75 and use_tired_training:
                train_phase = "add_kl"
            elif 0.75 <= epc / xlstm_cfg.epochs and use_tired_training:
                train_phase = "full_training"
        else:
            train_phase = None
        # train

        _ = train(
            model,
            optimizer,
            dataloader_train,
            device,
            compute_losses,
            writer,
            epc,
            train_avg,
            phase=train_phase,
        )

        best_results = valid(
            model,
            optimizer,
            dataloader_valid,
            device,
            compute_losses,
            writer,
            epc,
            valid_avg,
            phase=train_phase,
        )

        if best_results.grip_upsample_frame_pred is not None:
            grip_upsample_frame_pred_temp = best_results.grip_upsample_frame_pred
            grip_frame_labels_temp = best_results.grip_frame_labels
            side_upsample_frame_pred_temp = best_results.side_upsample_frame_pred
            side_frame_labels_temp = best_results.side_frame_labels

        # compute action loss std var and sem
        arr_act_mse_list = np.array(best_results.action_mse_list)
        best_results.act_std = arr_act_mse_list.std(ddof=0)
        best_results.act_var = arr_act_mse_list.var(ddof=0)
        # Avoid SmallSampleWarning: need at least 2 samples for sem
        if len(arr_act_mse_list) >= 2:
            best_results.act_sem = stats.sem(arr_act_mse_list, ddof=0)
        else:
            best_results.act_sem = 0.0
        arr_critic_mse_list = np.array(best_results.critic_mse_list)
        best_results.critic_std = arr_critic_mse_list.std(ddof=0)
        best_results.critic_var = arr_critic_mse_list.var(ddof=0)
        if len(arr_critic_mse_list) >= 2:
            best_results.critic_sem = stats.sem(arr_critic_mse_list, ddof=0)
        else:
            best_results.critic_sem = 0.0

        results_dict[f"epoch_{epc + 1}"] = {
            "train_actions_loss": train_avg.actions_loss.avg,
            "train_new_actions_loss": train_avg.new_actions_loss.avg,
            "train_grip_frames_loss": train_avg.grip_frames_loss.avg,
            "train_side_frames_loss": train_avg.side_frames_loss.avg,
            "train_image_kl_loss": train_avg.image_kl_loss.avg,
            "train_act_kl_loss": train_avg.act_kl_loss.avg,
            "train_sucker_loss": train_avg.sucker_loss.avg,
            "train_grip_diff_loss": train_avg.grip_diff_loss.avg,
            "train_side_diff_loss": train_avg.side_diff_loss.avg,
            "train_acc_sucker": train_avg.sucker_pred_acc.avg,
            ##
            "valid_actions_loss": valid_avg.actions_loss.avg,
            "valid_actions_std": best_results.act_std,
            "valid_actions_var": best_results.act_var,
            "valid_actions_sem": best_results.act_sem,
            "valid_new_actions_loss": valid_avg.new_actions_loss.avg,
            "valid_new_actions_std": best_results.critic_std,
            "valid_new_actions_var": best_results.critic_var,
            "valid_new_actions_sem": best_results.critic_sem,
            "valid_grip_frames_loss": valid_avg.grip_frames_loss.avg,
            "valid_side_frames_loss": valid_avg.side_frames_loss.avg,
            "valid_image_kl_loss": valid_avg.image_kl_loss.avg,
            "valid_act_kl_loss": valid_avg.act_kl_loss.avg,
            "valid_sucker_loss": valid_avg.sucker_loss.avg,
            "valid_grip_diff_loss": valid_avg.grip_diff_loss.avg,
            "valid_side_diff_loss": valid_avg.side_diff_loss.avg,
            "valid_acc_sucker": valid_avg.sucker_pred_acc.avg,
            "inference_time": valid_avg.infer_time.avg,
            "f1_score": best_results.f1_results["f1"],
            "precision": best_results.f1_results["precision"],
            "recall": best_results.f1_results["recall"],
            "act_jerk_mean": valid_avg.act_jerk_mean.avg,
            "act_jerk_norm": valid_avg.act_jerk_nrom.avg,
            "act_azc": valid_avg.act_azc.avg,
            "act_pi": valid_avg.act_pi.avg,
            "act_hf": valid_avg.act_hf.avg,
            "act_ea": valid_avg.act_ea.avg,
            "new_act_jerk_mean": valid_avg.new_act_jerk_mean.avg,
            "new_act_jerk_norm": valid_avg.new_act_jerk_nrom.avg,
            "new_act_azc": valid_avg.new_act_azc.avg,
            "new_act_pi": valid_avg.new_act_pi.avg,
            "new_act_hf": valid_avg.new_act_hf.avg,
            "new_act_ea": valid_avg.new_act_ea.avg,
            "label_act_jerk_mean": valid_avg.label_act_jerk_mean.avg,
            "label_act_jerk_norm": valid_avg.label_act_jerk_nrom.avg,
            "label_act_azc": valid_avg.label_act_azc.avg,
            "label_act_pi": valid_avg.label_act_pi.avg,
            "label_act_hf": valid_avg.label_act_hf.avg,
            "label_act_ea": valid_avg.label_act_ea.avg,
        }

        # ===================== WandB Logging (main process only) =====================
        if is_main_process() and use_wandb:
            wandb_log_dict = {
                "epoch": epc + 1,
                # Training metrics
                "train/actions_loss": train_avg.actions_loss.avg,
                "train/new_actions_loss": train_avg.new_actions_loss.avg,
                "train/grip_frames_loss": train_avg.grip_frames_loss.avg,
                "train/side_frames_loss": train_avg.side_frames_loss.avg,
                "train/image_kl_loss": train_avg.image_kl_loss.avg,
                "train/act_kl_loss": train_avg.act_kl_loss.avg,
                "train/sucker_loss": train_avg.sucker_loss.avg,
                "train/acc_sucker": train_avg.sucker_pred_acc.avg,
                # Validation metrics
                "valid/actions_loss": valid_avg.actions_loss.avg,
                "valid/actions_std": best_results.act_std,
                "valid/new_actions_loss": valid_avg.new_actions_loss.avg,
                "valid/grip_frames_loss": valid_avg.grip_frames_loss.avg,
                "valid/side_frames_loss": valid_avg.side_frames_loss.avg,
                "valid/image_kl_loss": valid_avg.image_kl_loss.avg,
                "valid/act_kl_loss": valid_avg.act_kl_loss.avg,
                "valid/sucker_loss": valid_avg.sucker_loss.avg,
                "valid/acc_sucker": valid_avg.sucker_pred_acc.avg,
                "valid/inference_time": valid_avg.infer_time.avg,
                # F1 metrics
                "valid/f1_score": best_results.f1_results["f1"],
                "valid/precision": best_results.f1_results["precision"],
                "valid/recall": best_results.f1_results["recall"],
            }
            wandb.log(wandb_log_dict)

        if best_results.acts_dict is not None:
            actions_prediction = {**actions_prediction, **best_results.acts_dict}
            new_actions_prediction = {
                **new_actions_prediction,
                **best_results.new_acts_dict,
            }
            # total_actions = {**total_actions, **best_results.total_actions}
        if not xlstm_cfg.olny_action_generate:
            metrix_results[f"epoch_{epc + 1}"] = {
                "grip_ssim_meam": valid_avg.grip_ssim_mean.avg,
                "grip_ssim_var": valid_avg.grip_ssim_var.avg,
                "grip_psnr_meam": valid_avg.grip_psnr_mean.avg,
                "grip_psnr_var": valid_avg.grip_psnr_var.avg,
                "side_ssim_meam": valid_avg.side_ssim_mean.avg,
                "side_ssim_var": valid_avg.side_ssim_var.avg,
                "side_psnr_meam": valid_avg.side_psnr_mean.avg,
                "side_psnr_var": valid_avg.side_psnr_var.avg,
            }
            #  ssim
            metrix_frames_result[f"epoch_{epc + 1}"] = {}

            metrix_grip_ssim_frame = {
                f"grip_ssim_frame_{f_cnt + 1}": f
                for f_cnt, f in enumerate(valid_avg.grip_ssim_frames.avg)
            }
            metrix_frames_result[f"epoch_{epc + 1}"].update(metrix_grip_ssim_frame)

            metrix_side_ssim_frame = {
                f"side_ssim_frame_{f_cnt + 1}": f
                for f_cnt, f in enumerate(valid_avg.side_ssim_frames.avg)
            }
            metrix_frames_result[f"epoch_{epc + 1}"].update(metrix_side_ssim_frame)

            metrix_grip_psnr_frame = {
                f"grip_psnr_frame_{f_cnt + 1}": f
                for f_cnt, f in enumerate(valid_avg.grip_psnr_frames.avg)
            }
            metrix_frames_result[f"epoch_{epc + 1}"].update(metrix_grip_psnr_frame)

            metrix_side_psnr_frame = {
                f"side_psnr_frame_{f_cnt + 1}": f
                for f_cnt, f in enumerate(valid_avg.side_psnr_frames.avg)
            }
            metrix_frames_result[f"epoch_{epc + 1}"].update(metrix_side_psnr_frame)

            if epc / xlstm_cfg.epochs >= (1.0 - xlstm_cfg.images_save_rate):
                best_results.grip_upsample_frame_pred = grip_upsample_frame_pred_temp
                best_results.grip_frame_labels = grip_frame_labels_temp
                best_results.side_upsample_frame_pred = side_upsample_frame_pred_temp
                best_results.side_frame_labels = side_frame_labels_temp
                for b in range(
                    xlstm_cfg.batchsize * xlstm_cfg.per_episode_to_batch_num
                ):
                    save_images(
                        best_results,
                        batch=b,
                        save_path=os.path.join(
                            pwd, xlstm_cfg.results_path, formatted_datetime
                        ),
                        epc=epc,
                        shape=xlstm_cfg.both_camera_concat_over,
                    )

    # ===================== Save results (main process only) =====================
    if is_main_process():
        writer = pd.ExcelWriter(
            os.path.join(
                pwd,
                xlstm_cfg.results_path,
                formatted_datetime,
                formatted_datetime + ".xlsx",
            ),
            engine="xlsxwriter",
        )
        results_pd = pd.DataFrame.from_dict(results_dict, orient="index")
        results_act = pd.DataFrame.from_dict(actions_prediction, orient="index")
        results_new_act = pd.DataFrame.from_dict(new_actions_prediction, orient="index")
        # total_actions_pd = pd.DataFrame.from_dict(total_actions, orient='index')
        metrix_frames_result_pd = pd.DataFrame.from_dict(
            metrix_frames_result, orient="index"
        )
        metrix_results_pd = pd.DataFrame.from_dict(metrix_results, orient="index")
        results_pd.reset_index(inplace=True)
        results_pd.rename(columns={"index": "Epochs"}, inplace=True)
        results_pd.to_excel(writer, sheet_name="results", index=False)
        results_act.to_excel(writer, sheet_name="actions", index=True)
        results_new_act.to_excel(writer, sheet_name="new_actions", index=True)
        # total_actions_pd.to_excel(writer, sheet_name='total_actions', index=True)
        metrix_frames_result_pd.to_excel(
            writer, sheet_name="frames_metrix_results", index=True
        )
        metrix_results_pd.to_excel(writer, sheet_name="metrix_results", index=True)
        # save_act(writer, actions_predict, actions_label, is_prediction=True)
        writer.close()
        OmegaConf.save(
            xlstm_cfg,
            os.path.join(
                pwd, xlstm_cfg.results_path, formatted_datetime, "config.yaml"
            ),
        )
        # Get model state dict (handle DDP wrapper)
        model_to_save = model.module if use_ddp else model
        save_checkpoint(
            {
                "epoch": epc + 1,
                "state_dict": model_to_save.state_dict(),
                # "model": model,   # Removed: sLSTM CUDA extensions are not picklable
                "optimizer": optimizer.state_dict(),
            },
            checkpoint="",
            filename=f"results/{formatted_datetime}/check_point/model_{formatted_datetime}.pth.tar",
        )

    # ===================== Cleanup =====================
    if is_main_process() and use_wandb:
        wandb.finish()
    if use_ddp:
        cleanup_ddp()


def train(
    model,
    optimizer,
    dataloader,
    device,
    compute_losses,
    writer,
    epc,
    train_avg,
    phase=None,
):
    seg_buffer = SegmentBuffer(config=xlstm_cfg)
    timestamp = 0
    model.train()

    pbar = tqdm(
        dataloader, desc=f"Train Epoch {epc + 1}/{xlstm_cfg.epochs}", leave=False
    )
    for batch_idx, batch in enumerate(pbar):
        # labels include: [sucker, joints_pos, gripper_frame, side_frame, joints_pos_with_timestamp]

        if xlstm_cfg.snn.is_use:
            functional.reset_net(model)

        # # test ##
        # from matplotlib import pyplot as plt
        # tttt = batch['action'][0, ...]
        # for i in range(tttt.shape[1]):
        #     plt.plot(tttt[:, i])
        #     plt.savefig(f'{i}.png')

        # for g in range(batch['obsetvation']['top_image'].shape[0]):
        #     for s in range(batch['obsetvation']['wrist_image'].shape[1]):
        #         plt.imshow(batch['obsetvation']['wrist_image'][g, s, ...].transpose(0, -1))
        #         plt.title(f'{g}_{s}')
        #         plt.savefig(f'{g}_{s}.png')
        ########

        if batch["action"].size(0) < xlstm_cfg.batchsize:
            continue

        batch = send_to_device(batch, device)

        # else:
        #     act_segments = seg_buffer.add_segment(timestamp_motors_sucker, add_side='l')

        _time = time.time()
        output = model(batch, phase)
        loss_results = compute_losses(output, batch, writer, epc, train_avg, phase)
        # compute acc
        pred_sucker = [
            torch.argmax(output.sucker[:, h, :], 1)
            for h in range(output.sucker.size(1))
        ]

        # Handle both LIBERO (no sucker_action) and JetMax (has sucker_action) datasets
        if "sucker_action" in batch:
            sucker_labels = batch["sucker_action"].cpu().data.numpy()
        else:
            # For LIBERO dataset, use zeros as placeholder (sucker loss is still computed separately in losses.py)
            sucker_labels = (
                torch.zeros(output.sucker.size(0), dtype=torch.long).cpu().data.numpy()
            )

        acc_gripper = sum(
            accuracy_score(
                pred_sucker[h].cpu().data.numpy(),
                sucker_labels,
            )
            for h in range(output.sucker.size(2))
        ) / (output.sucker.size(2))

        # pred_sucker = [torch.max(output.sucker[:, d, :], 1)[1] for d in range(xlstm_cfg.frames_seq)]

        # acc_sucker =  sum(accuracy_score(pred_sucker[d].cpu().data.numpy(), labels[0][:, d].data.numpy()) for d in range(xlstm_cfg.frames_seq)) / xlstm_cfg.frames_seq
        train_avg.sucker_pred_acc.update(acc_gripper)

        optimizer.zero_grad(set_to_none=True)
        loss_results.losses.backward()
        optimizer.step()
        _run_time = time.time() - _time

        # Update progress bar with metrics
        pbar.set_postfix(
            loss=f"{loss_results.losses.item():.2f}",
            gripper_acc=f"{acc_gripper:.4f}",
            time=f"{_run_time:.4f}",
        )
        # break
    return loss_results


def valid(
    model,
    optimizer,
    dataloader,
    device,
    compute_losses,
    writer,
    epc,
    valid_avg,
    seg_buffer=None,
    phase=None,
):
    model.eval()
    best_loss = 1e6
    timestamp = 0
    is_best_flag = None
    best_results = VariableContainer()
    seg_buffer = SegmentBuffer(config=xlstm_cfg)
    compute_ssimpsnr = MatrixSSIMPSNR()
    f1_metrix = TorchMetricsWrapper()
    pred_action, pred_new_act, label_action = [], [], []
    pred_grip_images, label_grip_images = [], []
    pred_side_images, label_side_images = [], []
    best_results.action_mse_list, best_results.critic_mse_list = [], []
    full_data_len = xlstm_cfg.batchsize * xlstm_cfg.per_episode_to_batch_num
    data_count = max(1, len(dataloader) // full_data_len)
    index = epc % data_count
    with torch.no_grad():
        pbar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"Valid Epoch {epc + 1}/{xlstm_cfg.epochs}",
            leave=False,
        )
        for batch_idx, batch_data in pbar:
            # Check if batch is in LIBERO dict format or JetMax tuple format
            if isinstance(batch_data, dict):
                # LIBERO batch dict format
                batch = send_to_device(batch_data, device)

                if batch["action"].size(0) < xlstm_cfg.batchsize:
                    continue

                if xlstm_cfg.snn.is_use:
                    functional.reset_net(model)

                _time = time.time()
                output = model(batch, phase)
                run_time = time.time() - _time

                loss_results = compute_losses(
                    output, batch, writer, epc, valid_avg, phase
                )

                best_results.action_mse_list.append(loss_results.actions_loss)
                best_results.critic_mse_list.append(loss_results.critic_loss)

                # compute acc - use placeholder for LIBERO (no gripper_action in labels)
                pred_gripper = torch.argmax(output.sucker, dim=-1)
                gripper_labels = torch.zeros(
                    output.sucker.size(0), dtype=torch.long, device=device
                )
                acc_gripper = 0.5  # Placeholder accuracy for LIBERO
                # Update F1 metrics for LIBERO
                f1_metrix.update(pred_gripper[:, -1], gripper_labels)

                valid_avg.sucker_pred_acc.update(acc_gripper)
                valid_avg.infer_time.update(run_time)

                # Collect predictions for metrics
                horizon = batch["action"].size(1)
                pred_new_act.append(
                    loss_results.new_actions[-1]
                    if loss_results.new_actions.dim() > 2
                    else loss_results.new_actions
                )
                pred_action.append(
                    output.actions[:, :horizon, :][-1]
                    if output.actions.dim() > 2
                    else output.actions[:, :horizon, :]
                )
                label_action.append(
                    batch["action"][-1]
                    if batch["action"].dim() > 2
                    else batch["action"]
                )

                # Update progress bar with metrics
                pbar.set_postfix(
                    loss=f"{loss_results.losses.item():.2f}",
                    gripper_acc=f"{acc_gripper:.4f}",
                    time=f"{run_time:.4f}",
                )

                # Break after first batch for validation (similar to training)
                # break
            else:
                # JetMax tuple format: (timestamp_motors_sucker, gripper_frame_timestamp, side_frame_timestamp, labels)
                (
                    timestamp_motors_sucker,
                    gripper_frame_timestamp,
                    side_frame_timestamp,
                    labels,
                ) = batch_data
                # labels include:  [0]sucker_labels, [1]joints_pos_labels, [2]gripper_frame_labels, [3]side_frame_labels
                labels = repeat_data(labels, xlstm_cfg.batchsize)
                timestamp_motors_sucker = repeat_data(
                    [timestamp_motors_sucker], xlstm_cfg.batchsize
                )[-1]
                gripper_frame_timestamp = repeat_data(
                    [gripper_frame_timestamp], xlstm_cfg.batchsize
                )[-1]
                side_frame_timestamp = repeat_data(
                    [side_frame_timestamp], xlstm_cfg.batchsize
                )[-1]
                gripper_frame_timestamp = gripper_frame_timestamp.transpose(2, -1)
                side_frame_timestamp = side_frame_timestamp.transpose(2, -1)
                labels[2] = labels[2].transpose(2, -1)
                labels[3] = labels[3].transpose(2, -1)

                if xlstm_cfg.snn.is_use:
                    functional.reset_net(model)

                if timestamp_motors_sucker.size(0) < xlstm_cfg.batchsize:
                    continue

                timestamp_motors_sucker = timestamp_motors_sucker.to(device=device)
                act_segments = seg_buffer.add_segment(timestamp_motors_sucker)

                gripper_frame_timestamp = gripper_frame_timestamp.to(device=device)
                side_frame_timestamp = side_frame_timestamp.to(device=device)
                labels = [label.to(device=device) for label in labels]

                output, run_time = inference_time(
                    model,
                    act_segments,
                    gripper_frame_timestamp,
                    side_frame_timestamp,
                    labels,
                )
                loss_results = compute_losses(output, labels, writer, epc, valid_avg)

                best_results.action_mse_list.append(loss_results.actions_loss)
                best_results.critic_mse_list.append(loss_results.critic_loss)

                # compute gripper action classification accuracy
                pred_gripper = torch.argmax(output.sucker, dim=-1)
                f1_metrix.update(pred_gripper[-1], labels[0][-1])
                acc_gripper = sum(
                    accuracy_score(
                        pred_gripper[:, h].cpu().data.numpy(),
                        labels[0][:, h].cpu().data.numpy(),
                    )
                    for h in range(output.sucker.size(1))
                ) / (output.sucker.size(1))
                valid_avg.sucker_pred_acc.update(acc_gripper)
                valid_avg.infer_time.update(run_time)
                if batch_idx in range(
                    (index) * full_data_len, (index + 1) * full_data_len
                ):
                    pred_new_act.append(loss_results.new_actions[-1])
                    pred_action.append(output.actions[-1])
                    label_action.append(labels[1][-1])
                    if (batch_idx + 1) % full_data_len == 0 and batch_idx > 0:
                        pred_act_dict = tens2act(pred_action, epc, name="pred")
                        pred_new_act_dict = tens2act(
                            pred_new_act, epc, name="pred_new_act"
                        )
                        label_act_dict = tens2act(label_action, epc, name="label")
                        calc_metrics(
                            torch.cat(pred_new_act, dim=0), valid_avg, flag="new_act"
                        )
                        calc_metrics(
                            torch.cat(pred_action, dim=0), valid_avg, flag="act"
                        )
                        calc_metrics(
                            torch.cat(label_action, dim=0), valid_avg, flag="label"
                        )

                        best_results.acts_dict = {**pred_act_dict, **label_act_dict}
                        best_results.new_acts_dict = {
                            **pred_new_act_dict,
                            **label_act_dict,
                        }
                        pred_new_act, pred_action, label_action = [], [], []

                    if not xlstm_cfg.olny_action_generate:
                        grip_results = compute_ssimpsnr.compute(
                            output.grip_future_seq, labels[2]
                        )
                        side_results = compute_ssimpsnr.compute(
                            output.side_future_seq, labels[3]
                        )
                        update_ssim_psnr(grip_results, side_results, valid_avg)

                        pred_grip_images.append(output.grip_future_seq_more[-1])
                        pred_side_images.append(output.side_future_seq_more[-1])
                        label_grip_images.append(labels[2][-1])
                        label_side_images.append(labels[3][-1])

                # Update progress bar with metrics
                pbar.set_postfix(
                    loss=f"{loss_results.losses.item():.2f}",
                    gripper_acc=f"{acc_gripper:.4f}",
                    time=f"{run_time:.4f}",
                )

        # if len(pred_action) == 1:
        #     pred_actions = pred_action[-1]
        #     label_actions = label_action[-1]
        # else:
        #     pred_actions = torch.cat(pred_action, dim=0)
        #     label_actions = torch.cat(label_action, dim=0)
        # pred_total_act_dict = tens2act(pred_actions, epc, name='pred_total_new_act')
        # label_total_act_dict = tens2act(label_actions, epc, name='label_total_act')
        # best_results.total_actions = {**pred_total_act_dict, **label_total_act_dict}
        best_results.total_actions = None

        if pred_grip_images and pred_side_images:
            best_results.grip_upsample_frame_pred = torch.cat(pred_grip_images, dim=0)
            best_results.grip_frame_labels = torch.cat(label_grip_images, dim=0)
            best_results.side_upsample_frame_pred = torch.cat(pred_side_images, dim=0)
            best_results.side_frame_labels = torch.cat(label_side_images, dim=0)
        else:
            best_results.grip_upsample_frame_pred = None
            best_results.grip_frame_labels = None
            best_results.side_upsample_frame_pred = None
            best_results.side_frame_labels = None

        best_results.actions_loss = loss_results.actions_loss
        best_results.critic_loss = loss_results.critic_loss
        best_results.f1_results = f1_metrix.compute()

        # Initialize acts_dict and new_acts_dict if not set (e.g., LIBERO dataset)
        if not hasattr(best_results, "acts_dict") or best_results.acts_dict is None:
            best_results.acts_dict = None
        if (
            not hasattr(best_results, "new_acts_dict")
            or best_results.new_acts_dict is None
        ):
            best_results.new_acts_dict = None

    return best_results


def init_weights(m):
    # 初始化所有线性层和卷积层（1D/2D/3D）
    if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        # 使用 Kaiming 初始化（假设使用 ReLU 激活）
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        # 初始化偏置为 0（如果存在）
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    # 初始化所有 BatchNorm 层（1D/2D/3D）
    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.ones_(m.weight)  # gamma 初始化为 1
        nn.init.zeros_(m.bias)  # beta 初始化为 0


def run_time(func):
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.time()
        with torch.no_grad():
            result = func(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        time_ = time.time() - start_time
        return result, time_

    return wrapper


def set_process_name_linux(new_name: str) -> None:
    if not new_name:
        return
    # Try setproctitle if available to change argv and process title as seen in ps
    try:
        import setproctitle  # type: ignore

        setproctitle.setproctitle(new_name)
    except Exception:
        pass
    # Set prctl PR_SET_NAME (affects /proc/<pid>/comm and default COMMAND in top)
    try:
        libc = ctypes.CDLL("libc.so.6")
        PR_SET_NAME = 15
        name_bytes = new_name.encode("utf-8")[:15]  # max 16 incl null
        libc.prctl(PR_SET_NAME, ctypes.c_char_p(name_bytes), 0, 0, 0)
    except Exception:
        pass
    # Best-effort write to /proc/self/comm as well
    try:
        with open("/proc/self/comm", "wb") as f:
            f.write((new_name[:15] + "\n").encode("utf-8"))
    except Exception:
        pass


@run_time
def inference_time(model, *args, **kwargs):
    return model(*args, **kwargs)


if __name__ == "__main__":
    # remarks that  when training on windows
    # if not rospy.core.is_initialized():
    #     rospy.init_node('act_generate', anonymous=True, log_level=rospy.DEBUG)
    parser = ArgumentParser()
    parser.add_argument("--config", default=r"./config.yaml")
    parser.add_argument(
        "--procname", default="python", help="设置进程名（Linux），用于 ps/top 显示"
    )
    parser.add_argument(
        "--use_ddp", action="store_true", help="启用分布式数据并行训练 (DDP)"
    )

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf8") as fp:
        config_yaml = fp.read()
    xlstm_cfg = OmegaConf.create(config_yaml)
    OmegaConf.resolve(xlstm_cfg)

    # 命令行参数覆盖配置文件
    if args.use_ddp:
        xlstm_cfg.use_ddp = True

    # 设置进程名（优先使用命令行，其次使用配置文件中的 process_name 字段）
    try:
        configured_name = xlstm_cfg.process_name if "process_name" in xlstm_cfg else ""
    except Exception:
        configured_name = ""
    procname = args.procname or configured_name
    if os.name == "posix" and procname:
        set_process_name_linux(procname)
    main(xlstm_cfg)

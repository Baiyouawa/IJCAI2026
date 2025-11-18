from models import TimesNet
import A_dataset
from torch import optim
import time
import numpy as np
import torch
import torch.nn as nn

# --- [新增] 引入 FGTI 公平的评估指标 ---
def calc_RMSE(target, forecast, eval_points):
    eval_p = torch.where(eval_points == 1)
    error_mean = torch.mean((target[eval_p] - forecast[eval_p])**2)
    return torch.sqrt(error_mean)

def calc_MAE(target, forecast, eval_points):
    eval_p = torch.where(eval_points == 1)
    error_mean = torch.mean(torch.abs(target[eval_p] - forecast[eval_p]))
    return error_mean

def calc_RSE(target, forecast, eval_points):
    """
    计算 RSE (相对平方误差)，只在 eval_points 为 1 的点上。
    """
    eval_p = torch.where(eval_points == 1)
    # 1. 仅提取“考题”的真实值和预测值
    true = target[eval_p]
    pred = forecast[eval_p]

    # 2. 计算分子 (预测误差的平方和)
    squared_error_num = torch.sum((true - pred)**2)

    # 3. 计算分母 (真实值与真实均值差异的平方和)
    true_mean = torch.mean(true)
    squared_error_den = torch.sum((true - true_mean)**2)
    
    # 4. 计算 RSE (添加 1e-5 避免除以零)
    rse_loss = squared_error_num / (squared_error_den + 1e-5)
    return rse_loss

def calc_RAE(target, forecast, eval_points):
    """
    计算 RAE (相对绝对误差)，只在 eval_points 为 1 的点上。
    """
    eval_p = torch.where(eval_points == 1)
    # 1. 仅提取“考题”的真实值和预测值
    true = target[eval_p]
    pred = forecast[eval_p]

    # 2. 计算分子 (预测误差的绝对值和)
    abs_error_num = torch.sum(torch.abs(true - pred))

    # 3. 计算分母 (真实值与真实均值差异的绝对值和)
    true_mean = torch.mean(true)
    abs_error_den = torch.sum(torch.abs(true - true_mean))
    
    # 4. 计算 RAE (添加 1e-5 避免除以零)
    rae_loss = abs_error_num / (abs_error_den + 1e-5)
    return rae_loss

# --- [新增结束] ---


def diffusion_train(configs):
    #从kdd.norm中读取，划分出train，test
    train_loader, test_loader = A_dataset.get_dataset(configs)
    model = TimesNet.Model(configs).to(configs.device)
    model_optim = optim.Adam(model.parameters(), lr=configs.learning_rate_diff, weight_decay=1e-6)
    p1 = int(0.75 * configs.epoch_diff)
    p2 = int(0.9 * configs.epoch_diff)
    #当炼丹跑到75%和90%时，把学习率降低10倍
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        model_optim, milestones=[p1, p2], gamma=0.1
    )
    loss_fn = nn.MSELoss()
    model.train()
    for epoch in range(configs.epoch_diff):
        iter_count = 0
        train_loss = []
        epoch_time = time.time()
        # model_optim.zero_grad() # (BUG: 应该在 Batch 循环内清零)
        for i, (observed_data_cpu, observed_dataf_cpu, observed_mask_cpu, observed_tp_cpu, gt_mask_cpu) in enumerate(train_loader):
            
            # --- [新增！] ---
            # 在这里把所有需要用到的张量从 CPU 移动到 GPU
            observed_data = observed_data_cpu.to(configs.device)
            observed_mask = observed_mask_cpu.to(configs.device)
            # gt_mask = gt_mask_cpu.to(configs.device) # (训练时不需要 gt_mask)
            # observed_dataf = observed_dataf_cpu.to(configs.device) # (训练时不应使用 dataf)

            iter_count += 1
            model_optim.zero_grad() # (BUG 修复：移到 Batch 循环内部)
            
            # --- [核心修复 1：动态“练习题”] ---
            # 1. (关键) 模拟 FGTI.get_randmask [cite: FGTI.py] 的逻辑
            #    我们只在“已知线索”(observed_mask == 1)的区域内随机挖洞
            
            # 1.1. 创建一个随机掩码 (0.0 -> 1.0)
            rand_for_mask = torch.rand_like(observed_mask)
        
            # 1.2. 定义“线索” (cond_mask)
            #      随机保留 50% 的“已知点”(observed_mask == 1) 作为线索
            cond_mask = ((observed_mask == 1) & (rand_for_mask > 0.5)).float()
        
            # 1.3. 定义“练习题” (target_mask)
            #      即 (所有已知点) - (我们用作线索的点)
            target_mask = observed_mask - cond_mask 

            # 1.4. 创建“带洞”的公平输入
            #      输入 x_enc 只包含“线索” (cond_mask)
            inp = observed_data * cond_mask

            # 2. 运行模型
            # (注意：传入 cond_mask，而不是 observed_mask)
            # (注意：不传入 x_mark_enc=None, x_dec=None, x_mark_dec=None)
            imputed_output = model(inp, None, None, None, mask=cond_mask)

            # 3. (关键) 只在“练习题” (target_mask) 上计算损失
            #    (我们不再使用 eval_mask，从而避免了“背诵”)
            loss = loss_fn(imputed_output[target_mask == 1], observed_data[target_mask == 1])
            
            # (处理 loss 可能为 NaN 的情况)
            if torch.isnan(loss):
                continue # 跳过这个 batch
            
            # --- [修复结束] ---

            loss.backward()
            model_optim.step()
            train_loss.append(loss.item())
            
        # (BUG 修复：lr_scheduler 必须在 Batch 循环【外部】调用)
        lr_scheduler.step()
        
        if epoch % 50 == 0 or epoch == configs.epoch_diff-1:
            train_loss = np.average(train_loss)
            print("Epoch: {}. Cost time: {}. Train_loss: {}.".format(epoch + 1, time.time() - epoch_time, train_loss))
    return model

def diffusion_test(configs, model):
    train_loader, test_loader = A_dataset.get_dataset(configs)
    model.eval()

    target_2d = []
    forecast_2d = []
    eval_p_2d = []
    generate_data2d = [] # <--- 我们把它加回来了，用于保存 .csv

    print("Testset sum: ", len(test_loader.dataset) // configs.batch + 1)

    start = time.time()
    # (注意：A_dataset [cite: A_dataset.py] 的 test_loader 和 train_loader 返回相同的 5-元组)
    for i, (observed_data_cpu, observed_dataf_cpu, observed_mask_cpu, observed_tp_cpu, gt_mask_cpu) in enumerate(test_loader):
        
        # --- [新增！] ---
        # 同样，在这里把所有需要用到的张量从 CPU 移动到 GPU
        observed_data = observed_data_cpu.to(configs.device)
        observed_mask = observed_mask_cpu.to(configs.device)
        gt_mask = gt_mask_cpu.to(configs.device)
        # observed_dataf = observed_dataf_cpu.to(configs.device) # (测试时不应使用 dataf)
        # --- [修改结束] ---

        # --- [核心修复 2：公平的测试输入] ---
        
        # 1. (关键) 创建“带洞”的公平输入
        #    我们使用 observed_mask (线索) 来掩码 observed_data (真实值)
        #    缺失点(mask=0)的值被设为 0
        x_enc = observed_data * observed_mask 
        
        with torch.no_grad():
            # 2. 运行模型
            # (注意：我们必须传入 observed_mask [cite: A_dataset.py] 作为 mask 参数，
            #  因为 TimesNet [cite: timesnet_model_user.py] 的 imputation [cite: timesnet_model_user.py] 函数需要它来进行“带掩码的标准化” [cite: timesnet_model_user.py])
            imputed_output = model(x_enc, None, None, None, mask=observed_mask)
            
        # 3. (关键) 计算“考题”掩码 (eval_mask)
        #    (这和 FGTI [cite: A_train_py_user.py] 的逻辑完全一致)
        eval_mask = gt_mask - observed_mask
        
        # --- [修复结束] ---

        imputed_sample = imputed_output.detach().to("cpu")
        observed_data = observed_data.detach().to("cpu")
        observed_mask = observed_mask.detach().to("cpu")
        gt_mask = gt_mask.detach().to("cpu")
        
        # "缝合"完整的数据
        imputed_data = observed_mask * observed_data + (1 - observed_mask) * imputed_sample
        evalmask = eval_mask.detach().to("cpu") # (eval_mask 之前在 GPU 上)

        target_2d.append(observed_data)
        forecast_2d.append(imputed_data)
        eval_p_2d.append(evalmask) 

        B, L, K = imputed_data.shape
        temp = imputed_data.reshape(B*L, K).numpy()
        generate_data2d.append(temp)

        #end = time.time()
        # (打印时间会拖慢测试，可以注释掉)
        # print("time cost for one batch:",end-start)
        #start = time.Eof() # (BUG: 应该是 time.time())
        #start = time.time() # (BUG 修复)


    generate_data2d = np.vstack(generate_data2d)
    np.savetxt(f"/mnt/4T/IJCAI/FGTI/Data/TimeNet_Imputation_{configs.dataset}_{configs.missing_rate}_{configs.seed}.csv", generate_data2d, delimiter=",")
    print(f"TimeNet imputation results saved to TimeNet_Imputation_{configs.dataset}_{configs.missing_rate}_{configs.seed}.csv") # 打印提示

    target_2d = torch.cat(target_2d, dim=0)
    forecast_2d = torch.cat(forecast_2d, dim=0)
    eval_p_2d = torch.cat(eval_p_2d, dim=0)

    # --- [使用公平的评估函数] ---
    RMSE = calc_RMSE(target_2d, forecast_2d, eval_p_2d)
    MAE = calc_MAE(target_2d, forecast_2d, eval_p_2d)
    RSE = calc_RSE(target_2d, forecast_2d, eval_p_2d)
    RAE = calc_RAE(target_2d, forecast_2d, eval_p_2d)


    print("RMSE: ", RMSE)
    print("MAE: ", MAE)
    print("RSE: ", RSE)
    print("RAE: ", RAE)
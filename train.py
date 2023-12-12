import os
import time
import torch
import wandb
import datetime
import warnings
from argparse import Namespace

warnings.filterwarnings("ignore")
from models.CTT_CGINet import CTT_CGINet
from utils.change_data import MyDataset
from utils.distributed_utils import set_seed
from utils.distributed_utils import ConfusionMatrix
from utils.train_and_eval import train_one_epoch, evaluate, create_lr_scheduler
import transforms as myTransforms


def save_checkpoint(state, filename):
    torch.save(state, filename)


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    batch_size = args.batch_size
    num_classes = args.num_classes + 1

    # 用来保存训练以及验证过程中信息
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # -------------------------训练前请修改mean,std-------------------------------#
    # # imagenet
    # mean = [0.485, 0.456, 0.406, 0.485, 0.456, 0.406] # 后三个时相2的mean,std
    # std = [0.229, 0.224, 0.225, 0.229, 0.224, 0.225]
    # # WHU train dataset
    # mean = [0.484, 0.463, 0.423, 0.484, 0.463, 0.423]
    # std = [0.194, 0.188, 0.201, 0.194, 0.188, 0.201]
    # LEVIR-CD
    mean = [0.398, 0.392, 0.335, 0.398, 0.392, 0.335]
    std = [0.187, 0.178, 0.166, 0.187, 0.178, 0.166]
    # # old_cdd
    # mean = [0.414,0.445,0.406,0.414,0.445,0.406]
    # std  = [0.229,0.247,0.233,0.229,0.247,0.233]

    trainDataset_main = myTransforms.Compose([
        myTransforms.Normalize(mean=mean, std=std),
        myTransforms.Scale(args.inWidth, args.inHeight),
        myTransforms.RandomCropResize(int(7. / 224. * args.inWidth)),
        myTransforms.RandomFlip(),
        myTransforms.RandomExchange(),
        myTransforms.ToTensor()
    ])

    valDataset = myTransforms.Compose([
        myTransforms.Normalize(mean=mean, std=std),
        myTransforms.Scale(args.inWidth, args.inHeight),
        myTransforms.ToTensor()
    ])

    train_dataset = MyDataset(args.data_path, transform=trainDataset_main)
    val_dataset = MyDataset(args.val_path, transform=valDataset)

    num_workers = 8  # todo: 默认为8
    print(num_workers)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=True
                                               )

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             num_workers=num_workers,
                                             pin_memory=True
                                             )
    model = CTT_CGINet(3, 2, args.pretrain, args.ckpt_path)
    model.to(device)

    if args.pretrain == True:  # 如果有预训练权重，则冻结住预训练权重
        for param in model.resnet.parameters():
            param.requires_grad = False
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs,
                                       warmup=True, warmup_epochs=5)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    if args.ckpt_url:
        print("使用预训练模型", args.ckpt_url)
        checkpoint = torch.load(args.ckpt_url, map_location='cpu')
        model.load_state_dict(checkpoint['model'])

    # 开始时间
    start_time = time.time()
    best_F1 = 0.
    Last_epoch = 0

    # 加载训练模型
    if args.resume_path and os.path.exists(args.resume_path):
        checkpoint = torch.load(args.resume_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        best_F1 = checkpoint['best_F1']
        Last_epoch = checkpoint['best_epoch']

    save_path = os.path.join("output", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(save_path)
    for epoch in range(args.start_epoch, args.epochs):
        mean_loss, lr = train_one_epoch(
            model, optimizer, train_loader, device, epoch,
            lr_scheduler=lr_scheduler,
            print_freq=args.print_freq,
            num_classes=num_classes,
            scaler=scaler)

        confmat = evaluate(model, val_loader,
                           device=device,
                           num_classes=num_classes, print_freq=args.print_freq)

        val_info = ConfusionMatrix.todict(confmat)
        val_info_print = str(confmat)
        # 各种评价指标
        precision = float(val_info['precision'][1])
        average_row_correct = float(val_info['average row correct'][1])
        Iou = float(val_info['IoU'][1])
        recall = float(val_info['recall'][1])
        Avg_precision = val_info['Avg_precision']
        F1 = float(val_info['F1_Score'][1])
        mean_Iou = val_info['mean IoU']

        print(val_info_print)
        if F1 == "nan":
            F1 = 0
        else:
            F1 = float(F1)
        save_txt = os.path.join(save_path, results_file)
        print(save_txt)
        with open(save_txt, "a") as f:
            # 记录每个epoch对应的train_loss、lr以及验证集各指标
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"lr: {lr:.6f}\n"

            f.write(train_info + val_info_print + "\n\n")
        if F1 > best_F1:
            best_F1 = F1
            Last_epoch = epoch
            model_name = "best_levir_new.pth"
            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optmizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'best_F1': best_F1,
                'best_epoch': Last_epoch
            }
            save_url = os.path.join(save_path, model_name)
            print(save_url)
            torch.save(checkpoint, save_url)

        print("Best:", best_F1)
        print("Best_epoch:", Last_epoch)
        wandb.init(project=args.project_name, config=args.__dict__, name='LEVIR-CD_(18,4.71)', save_code=True)
        wandb.log(
            {'epoch': epoch, 'F1': F1, 'precision': precision, 'IoU': Iou, 'recall': recall, 'mean_Iou': mean_Iou,
             'average_row_correct': average_row_correct, 'Avg_precision': Avg_precision, "lr": lr,
             "mean_loss": mean_loss, "best_F1": best_F1, "best_Epoch": Last_epoch})

        # 保存模型权重
        if epoch % 2 == 0:
            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'best_F1': best_F1,
                'best_epoch': Last_epoch
            }
            save_checkpoint(checkpoint, os.path.join(save_path, f'checkpoint_epoch_{epoch}.pth'))

    print("best model in {} epoch".format(Last_epoch))
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))

    # save code
    arti_code = wandb.Artifact('python', type='code')
    arti_code.add_file('./utils/change_data.py')
    arti_code.add_file('./utils/train_and_eval.py')
    arti_code.add_file('./models/CTT_CGINet.py')
    arti_code.add_file('train.py')
    wandb.log_artifact(arti_code)
    wandb.finish()


def parse_args():
    args = Namespace(
        project_name='ContrastModel',
        inWidth=256,
        inHeight=256,
        batch_size=32,
        pretrain=False,  # 是否采用遥感预训练权重
        ckpt_path='seco_resnet18_1m.ckpt',  # 遥感预训练权重
        data_path="../data/LEVIR-CD/train",
        val_path="../data/LEVIR-CD/val",
        out_path="./output",  # 权重checkpoint的保存地址
        device="cuda",
        num_classes=1,  # 除去背景的类别数目
        lr=4e-4,
        print_freq=100,
        epochs=100,
        start_epoch=0,
        save_path='checkpoint.pt',  # 暂时没有用上
        resume_path='',  # 恢复模型的位置
        ckpt_url="",
        amp=True,  # 采用混合精度加速，减少训练时间
        weight_decay=5e-4,
        seed=666)  # seed并不能保证每次结果一直，因为存在上采样等随机性操作
    return args


if __name__ == '__main__':
    args = parse_args()
    set_seed(args)
    main(args)

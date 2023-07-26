from utils import setup_argparser, compute_lambda_from_t, compute_t_from_lambda
from utils import compute_control_points_from_t_and_p, compute_p_from_control_points_and_t
from model import Model
from dataset import MyDataeset, MyDataeset_3

import numpy
import torch
from torch import nn, optim
from torch.utils.data import DataLoader


from torch.utils.tensorboard import SummaryWriter



def train_loop(dataloader : DataLoader, model : Model, loss_fn : nn.modules.loss._Loss, \
               optimizer : optim.Optimizer) -> float:
    total = len(dataloader.dataset)
    model.train()
    sum_loss, times = 0, 0
    for batch, (edges, label_t) in enumerate(dataloader):
        pred_lambdas = model(edges)

        # loss_fn2
        pred_t = compute_t_from_lambda(pred_lambdas)
        loss = loss_fn(pred_t, label_t)


        # loss_fn1
        # label_lambdas = compute_lambda_from_t(label_t)
        # loss = loss_fn(pred_lambdas, label_lambdas)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        sum_loss += loss.item()
        times += 1
        # if (batch+1) % 100 == 0 or (batch+1) == len(dataloader):
        #     loss, current = loss.item(), (batch + 1) * len(edges)
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{total:>5d}]")
    loss = sum_loss / times
    print(f"Train averages loss: {loss:>8f} \n")
    return loss


def val_loop(dataloader : DataLoader, model : Model, loss_fn : nn.modules.loss._Loss) -> float:
    num_batches = len(dataloader)
    loss = 0
    model.eval()
    with torch.no_grad():
        for edges, label_t in dataloader:
            pred_lambdas = model(edges)

            # loss_fn2
            pred_t = compute_t_from_lambda(pred_lambdas)       
            loss += loss_fn(pred_t, label_t).item() 


            # loss_fn1
            # label_lambdas = compute_lambda_from_t(label_t)
            # loss += loss_fn(pred_lambdas, label_lambdas).item()
    
    loss /= num_batches
    print(f"Test averages loss: {loss:>8f} \n")
    return loss


def main(args): 
    torch.set_default_dtype(torch.float64)
    torch.set_default_device(args.device)
    writer = SummaryWriter('logs')
    # 加载数据集
    data = numpy.load('data.npz')
    t =  torch.tensor(data['params'])    # (num_samples, 2*d+1)
    edges = torch.tensor(data['edges'])  # (num_samples, 2*d, 2)
    split_pos = int(edges.size(0) / args.batch_size * args.training_rate) * args.batch_size
    train_dataset = MyDataeset(edges[:split_pos], t[:split_pos])
    val_dataset = MyDataeset(edges[split_pos:], t[split_pos:])
    # print(len(train_dataset), len(val_dataset))
    train_dataloader = DataLoader(train_dataset, args.batch_size, True, generator=torch.Generator(device = args.device))
    val_dataloader = DataLoader(val_dataset, args.batch_size, True, generator=torch.Generator(device = args.device))
    # print(len(train_dataloader.dataset), len(val_dataloader.dataset))
    
    # 创建模型并定义损失函数和优化器
    model = Model(args.d)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr)
    
    # 训练并验证
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train_loss = train_loop(train_dataloader, model, loss_fn, optimizer)
        writer.add_scalar('train_loss', train_loss, epoch+1)
        val_loss = val_loop(val_dataloader, model, loss_fn)
        writer.add_scalar('val_loss', val_loss, epoch+1)
        # for para in model.parameters():
        #     print(para, end=' ')
        # print()
        if (epoch + 1) % 10 == 0:
            writer.flush()
            torch.save(model, 'model.pth')
            torch.save(model.state_dict(), 'model_params.pth')
        print("-------------------------------\n")
    print("Done!")
    writer.close()
    # 保存模型
    torch.save(model, 'model.pth')
    torch.save(model.state_dict(), 'model_params.pth')


def train_loop_3(dataloader : DataLoader, model : Model, loss_fn : nn.modules.loss._Loss, \
                 optimizer : optim.Optimizer) -> float:
    total = len(dataloader.dataset)
    model.train()
    sum_loss, times = 0, 0
    for batch, label_p in enumerate(dataloader):
        edges = label_p[:,1:,:] - label_p[:,:-1,:]
        pred_lambdas = model(edges)

        # loss_fn3
        pred_t = compute_t_from_lambda(pred_lambdas)
        pred_c = compute_control_points_from_t_and_p(pred_t, label_p)
        pred_p = compute_p_from_control_points_and_t(pred_c, pred_t)
        loss = loss_fn(pred_p, label_p)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        sum_loss += loss.item()
        times += 1
        # if (batch+1) % 100 == 0 or (batch+1) == len(dataloader):
        #     loss, current = loss.item(), (batch + 1) * len(edges)
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{total:>5d}]")
    loss = sum_loss / times
    print(f"Train averages loss: {loss:>8f} \n")
    return loss


def val_loop_3(dataloader : DataLoader, model : Model, loss_fn : nn.modules.loss._Loss) -> float:
    num_batches = len(dataloader)
    loss = 0
    model.eval()
    with torch.no_grad():
        for label_p in dataloader:
            edges = label_p[:,1:,:] - label_p[:,:-1,:]
            pred_lambdas = model(edges)

            # loss_fn3
            pred_t = compute_t_from_lambda(pred_lambdas)
            pred_c = compute_control_points_from_t_and_p(pred_t, label_p)
            pred_p = compute_p_from_control_points_and_t(pred_c, pred_t)    
            loss += loss_fn(pred_p, label_p).item() 

    loss /= num_batches
    print(f"Test averages loss: {loss:>8f} \n")
    return loss



def main_3(args):
    torch.set_default_dtype(torch.float64)
    torch.set_default_device(args.device)
    writer = SummaryWriter('logs')
    # 加载数据集
    data = numpy.load('data_3.npz')
    p = torch.tensor(data['points']) # (num_samples, 2*d+1, 2)
    split_pos = int(p.size(0) / args.batch_size * args.training_rate) * args.batch_size
    train_dataset = MyDataeset_3(p[:split_pos])
    val_dataset = MyDataeset_3(p[split_pos:])
    train_dataloader = DataLoader(train_dataset, args.batch_size, True, generator=torch.Generator(device = args.device))
    val_dataloader = DataLoader(val_dataset, args.batch_size, True, generator=torch.Generator(device = args.device))
    
    # 创建模型并定义损失函数和优化器
    model = Model(args.d)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr)
    
    # 训练并验证
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train_loss = train_loop_3(train_dataloader, model, loss_fn, optimizer)
        writer.add_scalar('train_loss', train_loss, epoch+1)
        val_loss = val_loop_3(val_dataloader, model, loss_fn)
        writer.add_scalar('val_loss', val_loss, epoch+1)
        if (epoch + 1) % 10 == 0:
            writer.flush()
            torch.save(model, 'model.pth')
            torch.save(model.state_dict(), 'model_params.pth')
        print("-------------------------------\n")
    print("Done!")
    writer.close()
    # 保存模型
    torch.save(model, 'model.pth')
    torch.save(model.state_dict(), 'model_params.pth')


if __name__ == '__main__':
    # 获取参数
    parser = setup_argparser()
    args = parser.parse_args()
    # main(args)
    main_3(args)
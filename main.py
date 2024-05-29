import os #看第55行
import sys
import time
import torch
import loguru
import pickle
import argparse
from loguru import logger
from loss import LossFunc
from model import AnchorModel
from dataloader import NeuproteinDataset, get_dataloader
from utils import set_seeds, get_distance_matrix_with_postprocess, get_distance_matrix_from_embeddings, get_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--seed', default=2023, type=str, help='random seed')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    parser.add_argument('--skip_train', action='store_false', help='train flag')
    parser.add_argument('--dataroot', default='./dataset', type=str, help='path to dataset')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--hidden_size', default=32, type=int, help='hidden size')
    parser.add_argument('--num_workers', default=4, type=int, help='num workers')
    parser.add_argument('--epochs', default=100, type=int, help='epochs')
    parser.add_argument('--eval_epochs', default=5, type=int, help='eval epochs')
    parser.add_argument('--t_max', default=10, type=int, help='t_max')
    parser.add_argument('--evaluate_topks', default=[1, 5, 10, 50, 100, 500], type=list, help='evaluate topks')
    parser.add_argument('--train_number', default=3000, type=int, help='number of training sequences')
    parser.add_argument('--query_number', default=500, type=int, help='number of query sequences')
    parser.add_argument('--model', default='baseline', type=str, help='model name')
    args = parser.parse_args()
    
    # set random seed
    topks = args.evaluate_topks
    set_seeds(args.seed)
    
    # load dataset
    train_seq_data_path, test_seq_data_path = os.path.join(args.dataroot, "train_protein_sequences.pkl"),\
                                              os.path.join(args.dataroot, "test_protein_sequences.pkl")
    print()
    train_dis_data_path, test_dis_data_path = os.path.join(args.dataroot, "train_protein_distances.pkl"),\
                                              os.path.join(args.dataroot, "test_protein_distances.pkl")
    
    train_sequences, test_sequences = pickle.load(open(train_seq_data_path, "rb")), \
                                      pickle.load(open(test_seq_data_path, "rb"))
    train_distance_matrixs, test_distance_matrixs = pickle.load(open(train_dis_data_path, "rb")), \
                                                    pickle.load(open(test_dis_data_path, "rb"))
    
    # make dataloader
    train_dataset = NeuproteinDataset(train_sequences)
    test_dataset = NeuproteinDataset(test_sequences)
    
    train_dataloader = get_dataloader(train_dataset, args.batch_size, args.num_workers)
    test_dataloader = get_dataloader(test_dataset, args.batch_size, args.num_workers, shuffle=False)
    
    # make model and train
    model = AnchorModel(args.hidden_size,args.model) #改成别的，就是baseline
    model.cuda()
    model = model.train()
    loss_function = LossFunc()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.t_max, eta_min=1e-5)
    
    # init metrics
    best_metrics = {'top%d' % topk: 0 for topk in topks}
    log_time = time.strftime("%Y-%m-%d %H_%M_%S", time.localtime())
    log_path = os.path.join("./log", "{}.log".format(log_time))
    #windows下的路径不能有冒号

    logger.add(log_path)
    logger.info("start training")
    logger.info("seed: {}".format(args.seed))
    
    from matplotlib import pyplot as plt
    plt.xlabel('step')
    plt.ylabel('loss')
    #数组用于存储loss
    loss_list = []
    val_list=[]
    for epoch in range(args.epochs):
        if args.skip_train:
            model = model.train()
            for (sequences,  
                sequence_masks,
                sequence_ids) in train_dataloader:
                sequences = sequences.cuda()
                sequence_masks = sequence_masks.cuda()
                protein_ids = [char for seen, char in enumerate(sequence_ids) if char not in sequence_ids[:seen]]
                protein_with_protein_target_distance = get_distance_matrix_with_postprocess(train_distance_matrixs, protein_ids, protein_ids)
                protein_with_protein_target_distance = protein_with_protein_target_distance.cuda()
                protein_with_protein_preidct_distance = model(sequences, sequence_masks)
                
                # calculate loss
                loss = loss_function(protein_with_protein_target_distance, 
                                     protein_with_protein_preidct_distance)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                best_metrics_str = " ".join(["{}: {}".format(key, value) for key, value in best_metrics.items()])
                logger.info("epoch: {}, loss: {}, metrics: {}".format(epoch, loss, best_metrics_str), sink=True)
                
                #加入loss
                loss_list.append(loss.detach().cpu().numpy())
                
                
            scheduler.step()
        
        
        plt.plot(loss_list,color='r',label='train_loss')
        plt.savefig('loss.png')
        model = model.eval()
        if epoch % args.eval_epochs == 0:
            model = model.eval()
            with torch.no_grad():
                target_distances = []
                predict_distances = []
                predict_protein_embeddings = []
                for (sequences,  
                    sequence_masks, 
                    sequence_ids) in test_dataloader:
                    sequences = sequences.cuda()
                    sequence_masks = sequence_masks.cuda()
                    protein_embeddings = model.forward_features(sequences, sequence_masks)
                    predict_protein_embeddings.append(protein_embeddings)
                    
                predict_protein_embeddings = torch.cat(predict_protein_embeddings, dim=0)
                
                query_protein_embeddings = predict_protein_embeddings[:args.query_number].cpu().numpy()
                gallery_protein_embeddings = predict_protein_embeddings[args.query_number:].cpu().numpy()
                
                predict_distance_matrix = get_distance_matrix_from_embeddings(query_protein_embeddings, gallery_protein_embeddings)
                target_distance_matrix = test_distance_matrixs[:args.query_number, args.query_number:]
                
                # calculate metrics
                metrics = get_metrics(predict_distance_matrix, target_distance_matrix, topks=topks)
                
                # update best metrics
                update_model_number = 0
                for topk in topks:
                    if metrics['top%d' % topk] > best_metrics['top%d' % topk]:
                        update_model_number += 1
                        
                if update_model_number > len(topks) // 2:
                    best_metrics = metrics
                    torch.save(model.state_dict(), "./checkpoint/anchor_model_{}.pth".format(epoch))
    
    #横坐标为训练次数，纵坐标为loss
    
    

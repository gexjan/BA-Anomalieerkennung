import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
import pandas as pd
import pickle
from model.lstm import LSTM
from util.modelmanager import save_model
from torch.utils.tensorboard import SummaryWriter
import time
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        if not torch.cuda.is_available():
            self.logger.warning("No CUDA available")
            torch.manual_seed(args.seed)
            self.kwargs = {}
            self.device = torch.device("cpu")
        else:
            self.kwargs = {'num_workers': 1, 'pin_memory': True}
            self.device = torch.device("cuda")
            self.logger.info('Using CUDA')
            torch.cuda.manual_seed(args.seed)

    def load_data(self):
        self.logger.info("Loading data")
        self.train_x = pd.read_pickle(os.path.join(self.args.data_dir, "{}_x.pkl".format((self.args.log_file).replace('.log', ''))))
        self.train_y = pd.read_pickle(os.path.join(self.args.data_dir, "{}_y.pkl".format((self.args.log_file).replace('.log', ''))))
        with open(os.path.join(self.args.data_dir, 'label_mapping.pkl'), 'rb') as f:
            self.label_mapping = pickle.load(f)

        self.num_classes = len(self.label_mapping)

    def create_dataloader(self):
        self.logger.info("Create Dataloader")

        X = torch.tensor(self.train_x['window'].tolist(), dtype=torch.float)
        Y = torch.tensor(self.train_y['next'].values)

        dataset = TensorDataset(X, Y)
        train_loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True, **self.kwargs)

        return train_loader


class DeeplogTrainer(Trainer):
    def train(self, train_loader):
        model = LSTM(self.args.input_size, self.args.hidden_size, self.args.num_layers, self.num_classes).to(self.device)

        lr = 0.001
        criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(model.parameters(), lr=lr)

        self.logger.info("Starting DeepLog training")
        log = 'adam_batch_size={}_epoch={}_log={}_layers={}_hidden={}_winsize={}_lr={}'.format(
            str(self.args.batch_size), str(self.args.epochs), self.args.log_file, self.args.num_layers,
            self.args.hidden_size, self.args.window_size, lr)
        writer = SummaryWriter(log_dir='log/' + log)

        total_step = len(train_loader)
        print("Steps: ", total_step)
        for epoch in range(self.args.epochs):
            model.train()
            train_loss = 0
            start_time = time.time()
            for step, (seq, label) in enumerate(train_loader):
                # Forward pass
                seq = seq.clone().detach().view(-1, self.args.window_size, self.args.input_size).to(self.device)
                output = model(seq)
                loss = criterion(output, label.to(self.device))

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
                writer.add_graph(model, seq)

            end_time = time.time()
            epoch_duration = end_time - start_time
            writer.add_scalar('train_loss', train_loss / total_step, epoch + 1)
            self.logger.debug('Epoch [{}/{}], train_loss: {:.4f}, time: {}'.format(epoch + 1, self.args.epochs,
                                                                                   train_loss / total_step,
                                                                                   epoch_duration))

            # self.logger.debug('Epoch [{}/{}], DeepLog Train_loss: {:.4f}, Time: {:.2f} sec'.format(
            #     epoch + 1, self.args.epochs, train_loss / len(train_loader.dataset), epoch_duration))

        save_model(model, self.args.input_size, self.args.hidden_size, self.args.num_layers, self.num_classes, self.args.model_dir)

        writer.close()


class AutoencoderTrainer(Trainer):
    def train(self):
        pass

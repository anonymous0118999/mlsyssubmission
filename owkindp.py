import time

import torch
import torch.nn.functional as F
from gradcnn import crb, make_optimizer
from torch import nn, optim

import data
import utils
from pytorch import get_data


class EmbeddingNet(crb.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        # Embedding dimension: vocab_size + <unk>, <pad>, <eos>, <sos>
        self.emb = crb.Embedding(vocab_size + 4, 16)
        self.pool = crb.AvgPool1d(256)
        self.fc1 = crb.Linear(16, 2)

    def forward(self, x, use_dp=False):
        x = self.emb(x)
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze()
        x = self.fc1(x)
        return x


class LSTMNet(crb.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        # Embedding dimension: vocab_size + <unk>, <pad>, <eos>, <sos>
        self.emb = crb.Embedding(vocab_size + 4, 100)
        self.lstm = crb.LSTM(100, 100)
        self.pool = crb.AvgPool1d(256)
        self.fc1 = crb.Linear(100, 2)

    def forward(self, x, use_dp=False):
        x = self.emb(x)
        x = x.transpose(0, 1)
        x, _ = self.lstm(x)
        x = x.transpose(0, 1)
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze()
        x = self.fc1(x)
        return x


class MNISTNet(crb.Module):
    def __init__(self, **_):
        super().__init__()
        self.conv1 = crb.Conv2d(1, 16, 8, 2, padding=3)
        self.conv2 = crb.Conv2d(16, 32, 4, 2)
        self.fc1 = crb.Linear(32 * 4 * 4, 32)
        self.fc2 = crb.Linear(32, 10)

    def forward(self, x):
        # x of shape [B, 1, 28, 28]
        x = F.relu(self.conv1(x))  # -> [B, 16, 14, 14]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 16, 13, 13]
        x = F.relu(self.conv2(x))  # -> [B, 32, 5, 5]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 32, 4, 4]
        x = x.view(-1, 32 * 4 * 4)  # -> [B, 512]
        x = F.relu(self.fc1(x))  # -> [B, 32]
        x = self.fc2(x)  # -> [B, 10]
        return x


class FFNN(crb.Module):
    def __init__(self, **_):
        super().__init__()
        self.fc1 = crb.Linear(104, 50)
        self.fc2 = crb.Linear(50, 2)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        return out


class Logistic(crb.Module):
    def __init__(self, **_):
        super().__init__()
        self.fc1 = crb.Linear(104, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = F.sigmoid(out)
        return out


model_dict = {
    'mnist': MNISTNet,
    'lstm': LSTMNet,
    'embed': EmbeddingNet,
    'ffnn': FFNN,
    'logreg': Logistic,
}


def main(args):
    print(args)
    assert args.dpsgd
    torch.backends.cudnn.benchmark = True

    train_data, train_labels = get_data(args)
    model = model_dict[args.experiment](vocab_size=args.max_features).cuda()
    model.get_detail(True)

    optimizer = make_optimizer(
        cls=optim.SGD,
        noise_multiplier=args.noise_multiplier,
        l2_norm_clip=args.l2_norm_clip,
    )(model.parameters(), lr=args.learning_rate)

    loss_function = nn.CrossEntropyLoss() if args.experiment != 'logreg' else nn.BCELoss()

    timings = []
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        dataloader = data.dataloader(train_data, train_labels, args.batch_size)
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            model.zero_grad()
            outputs = model(x)
            loss = loss_function(outputs, y)
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        duration = time.perf_counter() - start
        print("Time Taken for Epoch: ", duration)
        timings.append(duration)

    if not args.no_save:
        utils.save_runtimes(__file__.split('.')[0], args, timings)
    else:
        print('Not saving!')
    print('Done!')


if __name__ == '__main__':
    parser = utils.get_parser(model_dict.keys())
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="Noise multiplier (default 1.0)",
    )
    parser.add_argument(
        "-c",
        "--max-per-sample-grad_norm",
        type=float,
        default=1.0,
        help="Clip per-sample gradients to this norm (default 1.0)",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-5,
        help="Target delta (default: 1e-5)",
    )
    args = parser.parse_args()
    main(args)

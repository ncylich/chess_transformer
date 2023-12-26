from transformer import TransformerModel
from chess_loader import get_dataloader
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import math
import time
from datetime import timedelta

# TODO: enhance dataloader, implement gradient accumulation, test other optimizers
# TODO: test different hyperparameters

START_MODEL = None
END_MODEL = '1000_elo_engine'
NUM_GAMES = None

dropout = 0.2  # dropout is the probability of a given neuron's - typically between 0.2 and 0.5
LR = 0.01  # best around .0036 - .006 I believe
lr_type = 'normal'
num_epochs = 100

SEQ = 1
if SEQ:
    DATASET = 'labeled_elo_2200.txt'
    DATADICT = 'all_chess_move_ids.json'

    DOWNSIZE = 4  # @1 took 55 sec per epoch,@4 took 55 sec per epoch
    batch_size = 128  # hightest pow of 2 can handle is 64 (technically can do 80)
    accumulation_steps = 4

    d_model = 512  # dimensionality of embeddings
    nhead = 8
    nlayers = 6
    d_hid = 2048  # should be about 4 times d_model, can finetune later
    ntoken = 7773  # == 7772 + 1: Define the size of your vocabulary (number of unique chess moves)

    seq_len = 150  # 95th percentile of game moves (much quicker than going for biggest game size)
else:
    DATASET = "2200_10_k_pos.txt"
    DATADICT = 'board_pos_ids.json'

    DOWNSIZE = 16
    batch_size = 2048
    accumulation_steps = 1

    d_model = 512
    nhead = 2
    nlayers = 6
    d_hid = 2048
    ntoken = 15  # Define the size of your vocabulary (number of unique chess moves)

    seq_len = 129

start_time = time.time()
# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if DOWNSIZE != 1:
    d_model = int(d_model / DOWNSIZE)
    d_hid = int(d_hid / DOWNSIZE)


def train(model=None):
    if model is None:
        # Model Initialization
        # Typically all are powers of 2 like 512 or 1024, except nlayers and dropout
        model = TransformerModel(ntoken, d_model, nhead, d_hid, nlayers, dropout=dropout).to(device)
    else:
        model = torch.load(f'models/{START_MODEL}')

    # Loss Function and Optimizer
    loss_fn = nn.CrossEntropyLoss()

    '''
    torch.optim.lr_scheduler.StepLR: Decays the learning rate of each parameter group by a certain factor every specified number of epochs.
    torch.optim.lr_scheduler.ExponentialLR: Decays the learning rate of each parameter group by a gamma factor every epoch.
    torch.optim.lr_scheduler.ReduceLROnPlateau: Reduces learning rate when a metric has stopped improving.
    torch.optim.lr_scheduler.OneCycleLR: Sets the learning rate according to the 1cycle policy, which can be beneficial for training transformers.
    '''
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=(num_epochs//5), gamma=0.6)

    dataloader = get_dataloader(DATASET, DATADICT, batch_size=batch_size, sequence_len=seq_len, num_workers=0, num_games=NUM_GAMES)

    scaler = GradScaler()

    # Training Loop

    elapsed = str(timedelta(seconds=time.time() - start_time))
    print("Elapsed Time: ", elapsed)
    print('Training...')

    last_epoch = time.time()
    tot_train = 0
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0
        for batch, (inp, players, pieces, start_pos, end_pos, target) in enumerate(dataloader):
            inp, players, pieces, start_pos, end_pos, target = inp.to(device), players.to(device), pieces.to(device), start_pos.to(device), end_pos.to(device), target.to(device)  # casting to corr3ect device

            optimizer.zero_grad()
            with autocast():
                output = model(inp, players, pieces, start_pos, end_pos)
                # .view(-1, ntoken) reverses cols in torch matrices
                loss = loss_fn(output.view(-1, ntoken), target.view(-1))

            if (batch + 1) % accumulation_steps == 0 and str(loss.item()).lower() != 'nan':
                # Backward pass with scaled loss
                scaler.scale(loss).backward()
                # gradient clipping (float infinity errors)
                scaler.unscale_(optimizer)  # Unscale the gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients
                # Optimizer step with scaler
                scaler.step(optimizer)
                scaler.update()

            '''
            output = model(input)
            # .view(-1, ntoken) reverses cols in torch matrices
            loss = loss_fn(output.view(-1, ntoken), target.view(-1))
            loss.backward()
            optimizer.step()
            '''

            total_loss += loss.item()

        if not isinstance(lr_type, str) or lr_type != 'static':
            scheduler.step()  # updating learning rate after each epoch

        print(f'Epoch {epoch}: Loss {int(total_loss)} / {len(dataloader)} = {(total_loss / len(dataloader)):.{4}f}')
        elapsed = str(timedelta(seconds=time.time() - start_time))

        curr_epoch = time.time()
        delta_epoch = curr_epoch - last_epoch
        last_epoch = curr_epoch
        tot_train += delta_epoch
        avg_epoch = tot_train / epoch

        avg_epoch = str(timedelta(seconds=avg_epoch))
        print("Elapsed Time: ", elapsed, ', Average Generation Time: ', avg_epoch)

        if epoch % int(num_epochs / 10) == 0:
            torch.save(model, f'models/{END_MODEL}_{epoch}_whole.pth')

    torch.save(model, f'models/{END_MODEL}_whole.pth')


if __name__ == '__main__':
    train(model=START_MODEL)

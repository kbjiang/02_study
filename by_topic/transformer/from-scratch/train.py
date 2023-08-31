import os
import math
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from datasets import load_dataset
from tokenizers import Tokenizer
from transformer import (
    Transformer,
    PAD_IDX,
    BOS_IDX,
    EOS_IDX,
    device,
    block_size,
)
from timeit import default_timer as timer

SRC_LANG = 'en'
TGT_LANG = 'es'

batch_size = 32 # micro batch szie when accum_iter is greater than 1
accum_iter = 4
eval_iter = 2_000
num_epochs = 8
learning_rate = 3e-4
min_lr = 3e-5
lr_decay_iters = 200_000
warmup_iters = 20_000

log_file = 'logs.txt'

class token_encode:
    def __init__(self, lang, tokenizers):
        self.tokenizer = tokenizers[lang]

    def __call__(self, x):
        return self.tokenizer.encode(x).ids

# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# function to truncate list of tokens if over block size
def truncation_transform(token_ids):
    eid = min(len(token_ids), block_size-2)
    return token_ids[:eid]

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))

# function to collate data samples into batch tensors
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for sample in batch:
        src_batch.append(text_transform[SRC_LANG](sample[SRC_LANG].rstrip("\n")))
        tgt_batch.append(text_transform[TGT_LANG](sample[TGT_LANG].rstrip("\n")))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=True)
    return src_batch, tgt_batch

# https://github.com/karpathy/nanoGPT/blob/master/train.py
# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# https://pytorch.org/tutorials/beginner/translation_transformer.html
def train_epoch(model, optimizer, epoch):
    model.train()
    train_dataloader = DataLoader(train_iter, batch_size=batch_size, collate_fn=collate_fn)

    losses = 0
    epoch_time = 0
    start_time = timer()
    if epoch==0 and os.path.exists(log_file):
        os.remove(log_file)
    for batch_idx, (src, tgt) in enumerate(train_dataloader):
        src = src.to(device)
        tgt = tgt.to(device)
        _, loss = model(tgt[:, :-1], src, tgt[:, 1:])

        losses += loss.item()
        loss = loss / accum_iter
        loss.backward()

        # determine and set the learning rate for this iteration
        iter_num = epoch * len(train_dataloader) + batch_idx
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(train_dataloader)):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if ((batch_idx + 1) % eval_iter == 0):
            val_loss = evaluate(model)
            time = (timer() - start_time)
            epoch_time += time
            log = (
                f"Epoch: {epoch}, Batch: {batch_idx + 1}, "
                f"Train loss: {losses/eval_iter:.3f}, Val loss: {val_loss:.3f}, "
                f"LR: {lr: .3g}, Time: {time:.3f}s"
            )
            print(log)
            with open(log_file, 'a') as f:
                f.write(f"{log}\n")
            losses = 0
            start_time = timer()
    path = f"checkpoint-ep{epoch:02d}.pt"
    torch.save(model.state_dict(), path)
    print(f"Epoch: {epoch}, Time: {epoch_time//60}m; Saved path: {path}")

def evaluate(model):
    model.eval()
    losses = 0

    for src, tgt in val_dataloader:
        src = src.to(device)
        tgt = tgt.to(device)
        _, loss = model(tgt[:, :-1], src, tgt[:, 1:])

        losses += loss.item()
    return losses / len(list(val_dataloader))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    tokenizers = {}
    for lang in [SRC_LANG, TGT_LANG]:
        tokenizers[lang] = Tokenizer.from_file(f"tokenizer-{lang}.json")

    token_encodes = {}
    for lang in [SRC_LANG, TGT_LANG]:
        token_encodes[lang] = token_encode(lang, tokenizers)

    # ``src`` and ``tgt`` language text transforms to convert raw strings into tensors indices
    text_transform = {}
    for lang in [SRC_LANG, TGT_LANG]:
        text_transform[lang] = sequential_transforms(
            token_encodes[lang], #Tokenization
            truncation_transform,
            tensor_transform, # Add BOS/EOS and create tensor
        )

    print("load training dataset...")
    train_iter = load_dataset('opus100', language_pair='en-es', split='train')['translation']
    val_iter = load_dataset('opus100', language_pair='en-es', split='validation')['translation']
    train_dataloader = DataLoader(train_iter, batch_size=batch_size, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_iter, batch_size=batch_size, collate_fn=collate_fn)

    model = Transformer()
    model.to(device)

    print("Model loaded.")
    print(f"Total number of parameters in millions: {count_parameters(model)//1e6}")
    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print("start training...")
    for epoch in range(num_epochs):
        train_epoch(model, optimizer, epoch)

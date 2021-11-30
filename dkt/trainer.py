import os

import numpy as np
import torch
from sklearn.model_selection import KFold

from dkt.model.lstmattn import LSTMATTN

from .criterion import get_criterion
from .dataloader import Preprocess, get_loaders, partition_question
from .metric import get_metric
from .model import AKT, LSTM, SAINT, SAKT
from .optimizer import get_optimizer
from .scheduler import get_scheduler
from .utils import swa_init, swa_step, swap_swa_params
from logzero import logger


def run(args, data):
    if args.k_folds >= 2:
        kfold = map(
            lambda idx: (data[idx[0]], data[idx[1]]),
            KFold(n_splits=args.k_folds, shuffle=True).split(data),
        )
    else:
        kfold = [Preprocess.split_data(data, seed=args.seed)]

    for k, (train_data, valid_data) in enumerate(kfold):
        logger.info(f"Fold: {k + 1} / {args.k_folds}")

        if args.partition_question:
            train_data = partition_question(train_data, args)

        model_name = os.path.splitext(args.model_name)
        model_name = model_name[0] + f"_{k}" + model_name[1]

        train_loader, valid_loader = get_loaders(args, train_data, valid_data)

        # only when using warmup scheduler
        args.total_steps = int(len(train_loader.dataset) / args.batch_size) * (
            args.n_epochs
        )
        args.warmup_steps = args.total_steps // 10

        model = get_model(args)
        optimizer = get_optimizer(model, args)
        scheduler = get_scheduler(optimizer, args)

        best_auc = -1
        early_stopping_counter = 0
        swa_state = {}

        for epoch in range(args.n_epochs):
            logger.info(f"Start Training: Epoch {epoch + 1}")

            if epoch == args.swa_warmup:
                swa_init(model, swa_state)

            ### TRAIN
            train_auc, train_acc, train_loss = train(
                train_loader,
                model,
                optimizer,
                args,
            )

            swa_step(model, swa_state)
            swap_swa_params(model, swa_state)

            ### VALID
            auc, acc = validate(valid_loader, model, args)

            ### TODO: model save or early stopping
            # wandb.log(
            #     {
            #         "epoch": epoch,
            #         "train_loss": train_loss,
            #         "train_auc": train_auc,
            #         "train_acc": train_acc,
            #         "valid_auc": auc,
            #         "valid_acc": acc,
            #     }
            # )
            if auc > best_auc:
                best_auc = auc
                # torch.nn.DataParallel로 감싸진 경우 원래의 model을 가져옵니다.
                model_to_save = model.module if hasattr(model, "module") else model
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "state_dict": model_to_save.state_dict(),
                    },
                    args.model_dir,
                    model_name,
                )
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= args.patience:
                    logger.info(
                        f"EarlyStopping counter: {early_stopping_counter} out of {args.patience}"
                    )
                    break

            swap_swa_params(model, swa_state)

            # scheduler
            if args.scheduler == "plateau":
                scheduler.step(best_auc)
            else:
                scheduler.step()

    return best_auc


def train(train_loader, model, optimizer, args):
    model.train()

    total_preds = []
    total_targets = []
    losses = []
    for step, batch in enumerate(train_loader):
        if args.enable_da:
            aug_batch = augment_batch(tuple(b.clone() for b in batch))
            aug_input = process_batch(aug_batch, args)

        input = process_batch(batch, args)

        preds = model(input)
        targets = input[3]  # correct
        mask = input[4]

        loss = compute_loss(preds * mask, targets, args.compute_loss_only_last)

        if args.enable_da:
            aug_preds = model(aug_input)
            aug_targets = aug_input[3]  # correct
            aug_mask = aug_input[4]
            aug_loss = compute_loss(
                aug_preds * aug_mask, aug_targets, args.compute_loss_only_last
            )
            loss = (loss + aug_loss) / 2

        update_params(loss, model, optimizer, args)

        if step % args.log_steps == 0:
            logger.info(f"Training steps: {step} Loss: {str(loss.item())}")

        # predictions
        preds = preds[:, -1]
        targets = targets[:, -1]

        if args.device == "cuda":
            preds = preds.to("cpu").detach().numpy()
            targets = targets.to("cpu").detach().numpy()
        else:  # cpu
            preds = preds.detach().numpy()
            targets = targets.detach().numpy()

        total_preds.append(preds)
        total_targets.append(targets)
        losses.append(loss)

    total_preds = np.concatenate(total_preds)
    total_targets = np.concatenate(total_targets)

    # Train AUC / ACC
    auc, acc = get_metric(total_targets, total_preds)
    loss_avg = sum(losses) / len(losses)
    logger.info(f"TRAIN AUC : {auc} ACC : {acc}")
    return auc, acc, loss_avg


def validate(valid_loader, model, args):
    model.eval()

    total_preds = []
    total_targets = []
    for step, batch in enumerate(valid_loader):
        input = process_batch(batch, args, is_train=False)

        preds = model(input)
        targets = input[3]  # correct

        # predictions
        preds = preds[:, -1]
        targets = targets[:, -1]

        if args.device == "cuda":
            preds = preds.to("cpu").detach().numpy()
            targets = targets.to("cpu").detach().numpy()
        else:  # cpu
            preds = preds.detach().numpy()
            targets = targets.detach().numpy()

        total_preds.append(preds)
        total_targets.append(targets)

    total_preds = np.concatenate(total_preds)
    total_targets = np.concatenate(total_targets)

    # Train AUC / ACC
    auc, acc = get_metric(total_targets, total_preds)

    logger.info(f"VALID AUC : {auc} ACC : {acc}")

    return auc, acc


def inference(args, test_data):
    _, test_loader = get_loaders(args, None, test_data)
    preds_k = []
    for k in range(args.k_folds):
        model_name = os.path.splitext(args.model_name)
        model_name = model_name[0] + f"_{k}" + model_name[1]

        model = load_model(args, model_name)
        model.eval()

        total_preds = []
        for step, batch in enumerate(test_loader):
            input = process_batch(batch, args, is_train=False)

            preds = model(input)

            # predictions
            preds = preds[:, -1]

            if args.device == "cuda":
                preds = preds.to("cpu").detach().numpy()
            else:  # cpu
                preds = preds.detach().numpy()

            # total_preds += list(preds)
            total_preds.append(preds)
        preds_k.append(np.concatenate(total_preds))

    preds_k = np.stack(preds_k).mean(axis=0)

    write_path = os.path.join(args.output_dir, args.output_filename)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(write_path, "w", encoding="utf8") as w:
        w.write("id,prediction\n")
        for id, p in enumerate(preds_k):
            w.write("{},{}\n".format(id, p))


def get_model(args):
    """
    Load model and move tensors to a given devices.
    """
    if args.model == "lstm":
        model = LSTM(args)
    elif args.model == "lstmattn":
        model = LSTMATTN(args)
    elif args.model == "sakt":
        model = SAKT(args)
    elif args.model == "saint":
        model = SAINT(args)
    elif args.model == "akt":
        model = AKT(args)

    # if args.model == "lstmattn":
    #     model = LSTMATTN(args)
    # if args.model == "bert":
    #     model = Bert(args)

    model.to(args.device)

    return model


# Data augmentation
def augment_batch(batch):
    ridx = torch.randperm(batch[0].size(0))
    batch2 = tuple(b[ridx] for b in batch)

    new_batch = tuple(
        torch.cat([b1[:, -len(b1[0]) // 2 :], b2[:, -len(b2[0]) // 2 :]], axis=-1)
        for b1, b2 in zip(batch, batch2)
    )

    batch_size = len(new_batch)
    tuple_len = len(new_batch[0])
    seq_len = new_batch[0][0].shape[0]

    # zero padding 왼쪽으로 이동
    for b in range(batch_size):
        # new_batch[b][-1]: mask emb
        idx = (~new_batch[b][-1].bool()).nonzero(as_tuple=True)[0]
        if len(idx) > 0:
            idx_mask = torch.ones(seq_len, dtype=torch.bool)
            idx_mask[idx] = False
            new_idx = torch.cat([idx, idx_mask.nonzero(as_tuple=True)[0]])
            for i in range(tuple_len):
                new_batch[b][i] = new_batch[b][i][new_idx]

    return new_batch


# 배치 전처리
def process_batch(batch, args, is_train=True):

    test, question, tag, correct, mask = batch

    if args.random_permute and is_train:
        for i, q in enumerate(question):
            cnt_nonzero = q.count_nonzero()
            ridx = torch.randperm(cnt_nonzero)
            test[i, -cnt_nonzero:] = test[i, -cnt_nonzero:][ridx]
            question[i, -cnt_nonzero:] = question[i, -cnt_nonzero:][ridx]
            tag[i, -cnt_nonzero:] = tag[i, -cnt_nonzero:][ridx]
            correct[i, -cnt_nonzero:] = correct[i, -cnt_nonzero:][ridx]
            mask[i, -cnt_nonzero:] = mask[i, -cnt_nonzero:][ridx]

    # change to float
    mask = mask.type(torch.FloatTensor)
    correct = correct.type(torch.FloatTensor)

    if args.interaction_type in [0, 2]:
        #  interaction을 임시적으로 correct를 한칸 우측으로 이동한 것으로 사용
        #    saint의 경우 decoder에 들어가는 input이다
        interaction = correct + 1  # 패딩을 위해 correct값에 1을 더해준다.
        interaction = interaction.roll(shifts=1, dims=1)
        interaction[:, 0] = 0  # set padding index to the first sequence
        interaction = (interaction * mask).to(torch.int64)
    elif args.interaction_type == 3:
        interaction = (tag + 1) + correct * args.n_tag
        interaction = interaction.roll(shifts=1, dims=1)
        interaction[:, 0] = 0
        interaction = (interaction * mask).to(torch.int64)
    else:
        interaction = (question + 1) + correct * args.n_questions
        interaction = interaction.roll(shifts=1, dims=1)
        interaction[:, 0] = 0
        interaction = (interaction * mask).to(torch.int64)

    #  test_id, question_id, tag
    test = ((test + 1) * mask).to(torch.int64)
    question = ((question + 1) * mask).to(torch.int64)
    tag = ((tag + 1) * mask).to(torch.int64)

    # gather index
    # 마지막 sequence만 사용하기 위한 index
    gather_index = torch.tensor(np.count_nonzero(mask, axis=1))
    gather_index = gather_index.view(-1, 1) - 1

    # device memory로 이동

    test = test.to(args.device)
    question = question.to(args.device)

    tag = tag.to(args.device)
    correct = correct.to(args.device)
    mask = mask.to(args.device)

    interaction = interaction.to(args.device)
    gather_index = gather_index.to(args.device)

    return (test, question, tag, correct, mask, interaction, gather_index)


# loss계산하고 parameter update!
def compute_loss(preds, targets, only_last=False):
    """
    Args :
        preds   : (batch_size, max_seq_len)
        targets : (batch_size, max_seq_len)

    """
    loss = get_criterion(preds, targets)
    # 마지막 시퀀드에 대한 값만 loss 계산
    if only_last:
        loss = loss[:, -1]
    loss = torch.mean(loss)
    return loss


def update_params(loss, model, optimizer, args):
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
    optimizer.step()
    optimizer.zero_grad()


def save_checkpoint(state, model_dir, model_filename):
    logger.info("saving model ...")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(state, os.path.join(model_dir, model_filename))


def load_model(args, model_name):

    model_path = os.path.join(args.model_dir, model_name)
    logger.info(f"Loading Model from: {model_path}")
    load_state = torch.load(model_path)
    model = get_model(args)

    # 1. load model state
    model.load_state_dict(load_state["state_dict"], strict=True)

    logger.info(f"Loading Model from: {model_path} ...Finished.")
    return model

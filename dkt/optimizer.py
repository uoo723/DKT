from torch.optim import Adam, AdamW


def get_optimizer(model, args):
    if args.optimizer.lower() == "adam":
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=0.01)
    if args.optimizer.lower() == "adamw":
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    else:
        raise ValueError(f"{args.optimizer} is not supported")

    # 모든 parameter들의 grad값을 0으로 초기화
    optimizer.zero_grad()

    return optimizer

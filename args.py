import argparse


def parse_args(mode='train'):
    parser = argparse.ArgumentParser()


    parser.add_argument('--seed', default=42, type=int, help='seed')

    parser.add_argument('--device', default='gpu', type=str, help='cpu or gpu')

    parser.add_argument('--data_dir', default='data/', type=str, help='data directory')
    parser.add_argument('--asset_dir', default='data/', type=str, help='data directory')

    parser.add_argument('--file_name', default='train_data.csv', type=str, help='train file name')

    parser.add_argument('--model_dir', default='models/', type=str, help='model directory')
    parser.add_argument('--model_name', default='model.pt', type=str, help='model file name')

    parser.add_argument('--output_dir', default='output/', type=str, help='output directory')
    parser.add_argument('--output_filename', default='output.csv', type=str, help='output filename')
    parser.add_argument('--test_file_name', default='test_data.csv', type=str, help='test file name')

    parser.add_argument('--max_seq_len', default=20, type=int, help='max sequence length')
    parser.add_argument('--num_workers', default=1, type=int, help='number of workers')

    parser.add_argument('--partition_question', action='store_true',
        help='partition question if question length greater than max')

    # 0: padding - 0, incorrect - 1, correct: 2
    # 1: 2 * n_questions
    # 2: same size as option 0, fc(concat(question_emb;in_emb))
    parser.add_argument('--interaction_type', default=0, type=int, help='Set interaction type')
    parser.add_argument('--random_permute', action='store_true', help='random permute for seq. of data')

    # 모델
    parser.add_argument('--hidden_dim', default=64, type=int, help='hidden dimension size')
    parser.add_argument('--n_layers', default=2, type=int, help='number of layers')
    parser.add_argument('--n_heads', default=2, type=int, help='number of heads')
    parser.add_argument('--drop_out', default=0.2, type=float, help='drop out rate')

    # SAKT
    parser.add_argument('--attn_direction', default='uni', type=str,
        help='Set attention mask (uni|bi) direction')

    # AKT
    parser.add_argument('--l2', default=1e-5, type=float, help='l2 regularization')
    parser.add_argument('--final_fc_dim', default=512, type=int, help='hidden dim for final fc layer')
    parser.add_argument('--ff_dim', default=2048, type=int, help='dim for fc inside the basic block')

    # 훈련
    parser.add_argument('--n_epochs', default=20, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--clip_grad', default=10, type=float, help='clip grad')
    parser.add_argument('--patience', default=5, type=int, help='for early stopping')
    parser.add_argument('--swa-warmup', type=int, default=2, help='swa warmup')

    parser.add_argument('--compute_loss_only_last', action='store_true', help='only computes loss of last output')
    parser.add_argument('--k_folds', type=int, default=1, help='K-Fold validation')

    parser.add_argument('--log_steps', default=50, type=int, help='print log per n steps')

    # 데이터
    parser.add_argument('--enable_da', action='store_true', help='Enable naive data augmentation')

    ### 중요 ###
    parser.add_argument('--model', default='lstm', type=str, help='model type')
    parser.add_argument('--optimizer', default='adamw', type=str, help='optimizer type')
    parser.add_argument('--scheduler', default='plateau', type=str, help='scheduler type')

    args = parser.parse_args()

    return args

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument(
        "--epochs", type=int, default=100, help="number of rounds of training"
    )
    parser.add_argument("--num_users", type=int, default=10, help="number of users: K")
    parser.add_argument(
        "--split_ratio", type=float, default=0.75, help="the fraction of clients: C"
    )
    parser.add_argument(
        "--local_ep", type=int, default=10, help="the number of local epochs: E"
    )

    parser.add_argument(
        "--min_clust", type=int, default=2, help="the number of minimum cluster: O"
    )

    parser.add_argument(
        "--max_clust", type=int, default=9, help="the number of maximum cluster: M"
    )

    # other arguments
    parser.add_argument(
        "--iid", type=int, default=1, help="Default set to IID. Set to 0 for non-IID."
    )
    parser.add_argument(
        "--unequal",
        type=int,
        default=0,
        help="whether to use case 1 for  \
                        non-i.i.d setting (use 0 for case 2 (extream))",
    )

    # other adversarial arguments
    parser.add_argument(
        "--case",
        type=int,
        default=1,
        help="Default set to without_adversarial. Set to 0 for adversarial case.",
    )

    parser.add_argument(
        "--inject_adv", type=int, default=70, help="the round in which inject adversary"
    )

    parser.add_argument(
        "--gpu",
        default=None,
        help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.",
    )

    args = parser.parse_args()
    return args

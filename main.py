from option import args_parser
from fedlearning import FederatedLearning

# Usage Example
if __name__ == "__main__":
    args = args_parser()
    fl = FederatedLearning(
        args,
        num_clients=args.num_users,
        min_clusters=args.min_clust,
        max_clusters=args.max_clust,
        local_epochs=args.local_ep,
        class_split_ratio=args.split_ratio,
    )
    if args.case == "1":
        best_clustering = fl.train(args.epochs)
    else:
        best_clustering = fl.train(args.epochs, args.inject_adv)

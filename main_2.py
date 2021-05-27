import torch

import config
import utils
import trainer



def main(args):  # pylint:disable=redefined-outer-name
    """main: Entry point."""

    torch.manual_seed(args.random_seed)


    trnr = trainer.Trainer(args)

    trnr.train()

if __name__ == "__main__":
    args, unparsed = config.get_args()
    main(args)

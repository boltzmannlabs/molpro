from molpro.site_pred.model import SitePredModel
from argparse import ArgumentParser
import torch


def test_sitepred_model():
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--max_dist", type=int, default=35, help="maximum distance for atoms from the center of grid")
    parser.add_argument("--grid_resolution", type=int, default=2, help="resolution for the grid")
    parser.add_argument("--augment", type=bool, default=True, help="perform augmentation of dataset")
    parser.add_argument("--hdf_path", type=str, default="data.h5",
                        help="path where dataset is stored")
    parser.add_argument("--train_ids_path", type=str, default="sample_data/site_pred/splits/train.txt",
                        help="path where list of train ids is stored")
    parser.add_argument("--val_ids_path", type=str, default="sample_data/site_pred/splits/valid.txt",
                        help="path where list of validation ids is stored")
    parser.add_argument("--test_ids_path", type=str, default="sample_data/site_pred/splits/test.txt",
                        help="path where list of test ids is stored")
    parser.add_argument("--num_workers", type=int, default=8, help="number of workers for pytorch dataloader")
    parser.add_argument("--pin_memory", type=bool, default=True, help="whether to pin memory for pytorch dataloader")
    parser = SitePredModel.add_model_specific_args(parser)
    args = parser.parse_args()
    model = SitePredModel(**vars(args))
    print(model.hparams)
    x = torch.rand((2, 18, 36, 36, 36))
    preds = model(x)
    assert preds is not None and type(preds).__name__ == 'Tensor' and preds.shape == (2, 1, 36, 36, 36)

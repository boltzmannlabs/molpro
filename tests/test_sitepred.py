from boltpro.site_pred.model import SitepredModel, parser_args
import torch


def test_sitepred_model():
    x = torch.rand((2, 34, 36, 36, 36))
    hparams = parser_args()
    model = SitepredModel(hparams)
    preds = model(x)
    assert preds is not None and type(preds).__name__ == 'Tensor' and preds.shape == (2, 1, 36, 36, 36)

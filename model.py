from config import Config
import torch
import segmentation_models_pytorch as smp


class ModelWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        if Config.model_arch == 'FPN':
            arch = smp.FPN
        elif Config.model_arch == 'Unet':
            arch = smp.Unet
        elif Config.model_arch == 'DeepLabV3':
            arch = smp.DeepLabV3
        else:
            assert 0, f'Unknown architecture {Config.model_arch}'

        weights = 'imagenet' if Config.pretrained else None
        self.model = arch(
            encoder_name=Config.model_backbone, encoder_weights=weights, in_channels=Config.in_channels,
            classes=Config.num_classes, activation=None)

    def forward(self, x):
        x = self.model(x)
        return x


def test_model():
    model = ModelWrapper()
    x = torch.rand(2, 5, 128, 128)
    y = model(x)
    print(y.size())
    print(y)


if __name__ == '__main__':
    test_model()

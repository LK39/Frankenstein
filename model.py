import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision.models import EfficientNet_B0_Weights, ResNet18_Weights


def _adapt_head(model: nn.Module, num_classes: int = 10) -> nn.Module:
    """Adapt common torchvision model heads to `num_classes` (best-effort)."""
    # EfficientNet-style classifier (Sequential ending with Linear)
    if hasattr(model, 'classifier') and isinstance(model.classifier, (nn.Sequential,)):
        last = model.classifier[-1]
        if isinstance(last, nn.Linear):
            in_f = last.in_features
            model.classifier[-1] = nn.Linear(in_f, num_classes)
        else:
            model.classifier = nn.Sequential(nn.Linear(getattr(last, 'in_features', 1280), num_classes))
        return model

    # ResNet-style fc
    if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, num_classes)
        return model

    # Fallback: try to replace last Linear module found
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, nn.Linear):
            # don't attempt complex setattr; just leave as-is and warn
            return model

    return model


def load_pretrained(path: str, num_classes: int = 10, device: torch.device | None = None) -> nn.Module:
    """Load a pretrained model, adapt head to `num_classes`, and move to `device`.

    Special values for `path`:
      - 'efficientnet.pth' -> torchvision EfficientNet-B0 with ImageNet weights
      - 'biasnn.pth' -> torchvision ResNet18 (placeholder for BIASNN)
    If `path` is a real file path, it will be loaded with `torch.load`.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if path == 'efficientnet.pth':
        try:
            weights = EfficientNet_B0_Weights.DEFAULT
            model = models.efficientnet_b0(weights=weights)
        except Exception:
            model = models.efficientnet_b0(pretrained=True)
    elif path == 'biasnn.pth':
        try:
            weights = ResNet18_Weights.DEFAULT
            model = models.resnet18(weights=weights)
        except Exception:
            model = models.resnet18(pretrained=True)
    else:
        model = torch.load(path)

    model = _adapt_head(model, num_classes=num_classes)
    model.to(device)
    model.eval()
    model.summary() if hasattr(model, 'summary') else None
    return model


class HybridModel(nn.Module):
    """Hybrid model that stitches the first `cut_point` children of model A
    with the remaining children of model B, adding an adapter if needed.
    """

    def __init__(self, model_a_path: str, model_b_path: str, cut_point: int, input_shape=(3, 224, 224), device: torch.device | None = None):
        super().__init__()
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model_a = load_pretrained(model_a_path, device=device)
        model_b = load_pretrained(model_b_path, device=device)

        children_a = list(model_a.children()) 
        children_b = list(model_b.children())

        self.encoder = nn.Sequential(*children_a[:cut_point])
        self.decoder = nn.Sequential(*children_b[cut_point:])
        self.encoder.to(device)
        self.decoder.to(device)

        # snapshot device
        self.device = device

        # determine encoder output shape with a dummy input
        dummy = torch.randn(1, *input_shape, device=device)
        with torch.no_grad():
            enc_out = self.encoder(dummy)

        # encoder output info
        self._enc_ndim = enc_out.ndim
        if self._enc_ndim > 2:
            # keep channels and spatial info
            self._enc_channels = enc_out.shape[1]
            self._enc_spatial = tuple(enc_out.shape[2:])
            # flattened dim if needed
            enc_flat = enc_out.view(enc_out.size(0), -1)
            enc_dim = enc_flat.shape[1]
            self.flatten = False  # do NOT flatten 4D outputs by default
        else:
            self._enc_channels = None
            self._enc_spatial = None
            enc_dim = enc_out.shape[1]
            self.flatten = True

        # Inspect decoder's first layer to decide adapter type
        dec_first = None
        try:
            if len(self.decoder) > 0:
                dec_first = self.decoder[0]
        except Exception:
            dec_first = None

        self.adapter = None
        # If decoder expects Conv2d input, prefer a Conv2d 1x1 adapter (preserve spatial dims)
        if isinstance(dec_first, nn.Conv2d):
            # If encoder gives 4D output, map channels -> expected in_channels
            if self._enc_channels is not None:
                enc_ch = self._enc_channels
                dec_in_ch = dec_first.in_channels
                if enc_ch != dec_in_ch:
                    self.adapter = nn.Conv2d(enc_ch, dec_in_ch, kernel_size=1).to(device)
            else:
                # encoder produced flattened vector but decoder expects conv input
                # we'll map flattened vector to channels and add spatial 1x1
                dec_in_ch = dec_first.in_channels
                self.adapter = nn.Linear(enc_dim, dec_in_ch).to(device)
                # mark that we need to unflatten after linear adapter
                self._unflatten_after_linear = True

        # If decoder expects a Linear input (fully-connected), ensure adapter maps flattened dim
        elif isinstance(dec_first, nn.Linear):
            dec_in = dec_first.in_features
            if enc_dim != dec_in:
                self.adapter = nn.Linear(enc_dim, dec_in).to(device)

        else:
            # best-effort: if dims mismatch, add a Linear adapter between flattened enc and decoder
            try:
                dec_dim = enc_dim
                if hasattr(dec_first, 'in_features'):
                    dec_dim = dec_first.in_features
                if enc_dim != dec_dim:
                    self.adapter = nn.Linear(enc_dim, dec_dim).to(device)
            except Exception:
                self.adapter = None

        # Try a dry-run through adapter+decoder with the dummy to ensure compatibility.
        # If incompatible (common when stitching very different backbones), fall back
        # to a simple classifier decoder (Flatten -> Linear) so hybrid can still run.
        try:
            with torch.no_grad():
                test = enc_out
                if self.adapter is not None:
                    if isinstance(self.adapter, nn.Conv2d):
                        test = self.adapter(test)
                    else:
                        # linear adapter expects flat input
                        if test.ndim > 2:
                            test = test.view(test.size(0), -1)
                        test = self.adapter(test)
                        if getattr(self, '_unflatten_after_linear', False):
                            test = test.unsqueeze(-1).unsqueeze(-1)
                _ = self.decoder(test)
        except Exception:
            # Fallback: replace decoder with a simple classifier so hybrid is usable.
            num_classes = 10
            try:
                if hasattr(model_b, 'fc') and isinstance(model_b.fc, nn.Linear):
                    num_classes = model_b.fc.out_features
                elif hasattr(model_b, 'classifier') and isinstance(model_b.classifier, (nn.Sequential,)):
                    last = model_b.classifier[-1]
                    if isinstance(last, nn.Linear):
                        num_classes = last.out_features
            except Exception:
                pass

            # Ensure we map flattened encoder output to classifier input
            self.adapter = None
            self.decoder = nn.Sequential(nn.Flatten(1), nn.Linear(enc_dim, num_classes)).to(device)

        # Ensure an adapter module exists so there's always something small to train.
        # This guarantees `train_only` can enable adapter parameters and avoid the
        # "no trainable parameters" situation.
        if self.adapter is None:
            try:
                if isinstance(dec_first, nn.Conv2d):
                    if self._enc_channels is not None:
                        self.adapter = nn.Conv2d(self._enc_channels, dec_first.in_channels, kernel_size=1).to(device)
                    else:
                        # map flattened -> channels then unflatten
                        self.adapter = nn.Linear(enc_dim, dec_first.in_channels).to(device)
                        self._unflatten_after_linear = True
                elif isinstance(dec_first, nn.Linear):
                    dec_in = dec_first.in_features
                    self.adapter = nn.Linear(enc_dim, dec_in).to(device)
                else:
                    # Best-effort linear adapter
                    dec_dim = enc_dim
                    if hasattr(dec_first, 'in_features'):
                        dec_dim = dec_first.in_features
                    if enc_dim != dec_dim:
                        self.adapter = nn.Linear(enc_dim, dec_dim).to(device)
            except Exception:
                # leave adapter as None if creation fails
                self.adapter = None

        # Debugging info: small print to indicate adapter creation (helpful during runs)
        try:
            if self.adapter is not None:
                param_count = sum(p.numel() for p in self.adapter.parameters())
                print(f'HybridModel: adapter created: {type(self.adapter).__name__}, params={param_count}')
            else:
                print('HybridModel: no adapter created (unexpected)')
        except Exception:
            pass

    def forward(self, x):
        x = x.to(self.device)
        x = self.encoder(x)

        # If encoder produced 4D and decoder expects Conv2d, keep 4D and apply conv adapter if present
        if x.ndim > 2:
            if self.adapter is not None:
                # Conv2d adapter expects 4D input
                if isinstance(self.adapter, nn.Conv2d):
                    x = self.adapter(x)
                else:
                    # linear adapter: flatten, map, then unflatten to (B, C, 1, 1)
                    x = x.view(x.size(0), -1)
                    x = self.adapter(x)
                    if getattr(self, '_unflatten_after_linear', False):
                        # assume adapter output is channels; shape to (B, C, 1, 1)
                        x = x.unsqueeze(-1).unsqueeze(-1)
            # else: pass 4D through decoder (decoder's first layer should handle 4D)
        else:
            # encoder gave 2D output
            if self.adapter is not None:
                x = self.adapter(x)
            else:
                # if decoder expects conv input, try to reshape to (B, C, 1, 1)
                try:
                    first_layer = self.decoder[0]
                    if isinstance(first_layer, nn.Conv2d):
                        # best-effort: treat second dim as channels
                        x = x.unsqueeze(-1).unsqueeze(-1)
                except Exception:
                    pass

        x = self.decoder(x)
        return x


def freeze(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_all(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = True


def train_only(model: HybridModel, adapter: nn.Module | None):
    # Freeze encoder and decoder by default
    freeze(model.encoder)
    freeze(model.decoder)

    # If an explicit adapter exists, enable only its parameters
    if adapter is not None:
        for p in adapter.parameters():
            p.requires_grad = True
        return

    # No explicit adapter: try to find a reasonable "connecting" layer in decoder
    # and enable only its parameters (e.g., last Linear in decoder or first Linear)
    try:
        # prefer first Linear (after possible Flatten)
        for module in model.decoder.modules():
            if isinstance(module, nn.Linear):
                for p in module.parameters():
                    p.requires_grad = True
                return
        # fallback: if decoder is a Sequential and its last module is Linear
        if isinstance(model.decoder, nn.Sequential) and len(model.decoder) > 0:
            last = model.decoder[-1]
            if isinstance(last, nn.Linear):
                for p in last.parameters():
                    p.requires_grad = True
                return
    except Exception:
        pass

    # As a last resort, enable any parameter in decoder (least preferred)
    for p in model.decoder.parameters():
        p.requires_grad = True


def train_entire_model(model: nn.Module):
    unfreeze_all(model)


# Dataset helpers (create datasets/loaders lazily to avoid download side-effects at import)
def make_default_transform(resize: int = 224):
    return transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def get_cifar10_loader(root: str = './data', train: bool = True, batch_size: int = 64, shuffle: bool = True, download: bool = True, transform=None):
    if transform is None:
        transform = make_default_transform()
    ds = CIFAR10(root=root, train=train, download=download, transform=transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device | None = None) -> float:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            if outputs.ndim == 1:
                # single-value output: skip
                continue
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total if total > 0 else 0.0


def train_model(model: nn.Module, epochs: int = 1, dataloader: DataLoader | None = None, device: torch.device | None = None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    if dataloader is None:
        dataloader = get_cifar10_loader()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        print('No trainable parameters found (all parameters are frozen). Skipping training.')
        return
    optimizer = torch.optim.Adam(trainable_params, lr=0.001)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader):.4f}")


def to_device(model: nn.Module, device: torch.device):
    model.to(device)


def move_batch_to_device(inputs, labels, device: torch.device):
    return inputs.to(device), labels.to(device)

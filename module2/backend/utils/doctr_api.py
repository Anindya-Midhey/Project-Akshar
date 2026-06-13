import os
import cv2
import numpy as np

# Global variables to hold the loaded model and PyTorch references
_DOCTR_MODEL = None
_TORCH = None
_F = None

def _lazy_init():
    """Lazy load PyTorch and DocTr weights to avoid massive startup times."""
    global _DOCTR_MODEL, _TORCH, _F
    if _DOCTR_MODEL is not None:
        return

    import torch
    import torch.nn.functional as F
    _TORCH = torch
    _F = F
    
    # Import from our deps folder
    from .doctr_deps.seg import U2NETP
    from .doctr_deps.GeoTr import GeoTr
    
    class GeoTr_Seg(torch.nn.Module):
        def __init__(self):
            super(GeoTr_Seg, self).__init__()
            self.msk = U2NETP(3, 1)
            self.GeoTr = GeoTr(num_attn_layers=6)

        def forward(self, x):
            msk, _1, _2, _3, _4, _5, _6 = self.msk(x)
            msk = (msk > 0.5).float()
            x = msk * x
            bm = self.GeoTr(x)
            bm = (2 * (bm / 286.8) - 1) * 0.99
            return bm

    def reload_model(model, path=""):
        model_dict = model.state_dict()
        pretrained_dict = torch.load(path, map_location='cpu')
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        return model

    def reload_segmodel(model, path=""):
        model_dict = model.state_dict()
        pretrained_dict = torch.load(path, map_location='cpu')
        pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if k[6:] in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        return model

    # Instantiate model
    model = GeoTr_Seg()
    
    # Resolve absolute paths to models
    base_dir = os.path.dirname(os.path.dirname(__file__))
    models_dir = os.path.join(base_dir, "models", "doctr")
    
    seg_path = os.path.join(models_dir, "seg.pth")
    geotr_path = os.path.join(models_dir, "geotr.pth")
    
    if not os.path.exists(seg_path) or not os.path.exists(geotr_path):
        raise FileNotFoundError(f"DocTr weights missing from {models_dir}")
        
    reload_segmodel(model.msk, seg_path)
    reload_model(model.GeoTr, geotr_path)
    
    model.eval()
    
    # Optionally configure to use CUDA/MPS if available, but stick to CPU for safety unless user needs speed.
    # Currently forcing CPU as it's safe and doesn't require PyTorch compilation fixes on Mac.
    
    _DOCTR_MODEL = model

def apply_doctr_dewarp(image: np.ndarray) -> np.ndarray:
    """
    Applies Deep Learning Geometric Unwarping.
    Initializes the 150MB+ PyTorch model payload gracefully into memory on the first run.
    """
    _lazy_init()
    
    # Prepare image
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
    im_ori = image / 255.0
    h, w, _ = im_ori.shape
    im = cv2.resize(im_ori, (288, 288))
    im = im.transpose(2, 0, 1)
    im_tensor = _TORCH.from_numpy(im).float().unsqueeze(0)
    
    with _TORCH.no_grad():
        bm = _DOCTR_MODEL(im_tensor)
        bm = bm.cpu()
        bm0 = cv2.resize(bm[0, 0].numpy(), (w, h))
        bm1 = cv2.resize(bm[0, 1].numpy(), (w, h))
        
        # Blur mapping
        bm0 = cv2.blur(bm0, (3, 3))
        bm1 = cv2.blur(bm1, (3, 3))
        lbl = _TORCH.from_numpy(np.stack([bm0, bm1], axis=2)).unsqueeze(0)
        
        # Apply transformation
        im_ori_tensor = _TORCH.from_numpy(im_ori).permute(2, 0, 1).unsqueeze(0).float()
        out = _F.grid_sample(im_ori_tensor, lbl, align_corners=True)
        
        out_image = ((out[0] * 255).permute(1, 2, 0).numpy())
        
        # DocTr often predicts slightly beyond bounds resulting in floats - clamp safely
        img_geo = np.clip(out_image, 0, 255).astype(np.uint8)
        
    return img_geo



i use same code for enhancement in module 1 and 2 
but in module 1 it shows good resuls but in module 2 work bench it give worst result on same enhance what is the reason

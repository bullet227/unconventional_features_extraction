# features/image_features.py
"""
CNN-based candlestick pattern recognition using ResNet18.
Requires PyTorch, torchvision, and OpenCV.
"""
import numpy as np
import polars as pl
import warnings
import logging

log = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Try to import GPU/CV dependencies
try:
    import cv2
    import torch
    from torchvision.models import resnet18, ResNet18_Weights
    from torchvision import transforms

    CV_AVAILABLE = True

    # Use lightweight model and extract features before final layer
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _base_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(_device).eval()
    # Remove final classification layer to get 512-dim features
    _model = torch.nn.Sequential(*list(_base_model.children())[:-1])

    _tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

except ImportError as e:
    CV_AVAILABLE = False
    _device = None
    _model = None
    _tf = None
    log.warning(f"Image features unavailable (missing dependencies: {e}). "
                "Install with: pip install torch torchvision opencv-python")

def create_candle_pattern_image(ohlc_window, img_size=(224, 224)):
    """Create a visual representation of candlestick patterns."""
    if not CV_AVAILABLE:
        return np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)

    img = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
    
    if len(ohlc_window) == 0:
        return img
    
    # Normalize prices to fit in image
    high_max = ohlc_window[:, 1].max()
    low_min = ohlc_window[:, 2].min()
    price_range = high_max - low_min + 1e-8
    
    # Draw candlesticks
    candle_width = img_size[1] // len(ohlc_window)
    
    for i, (o, h, l, c) in enumerate(ohlc_window):
        x = i * candle_width + candle_width // 2
        
        # Normalize to image coordinates
        o_y = int((1 - (o - low_min) / price_range) * (img_size[0] - 20) + 10)
        h_y = int((1 - (h - low_min) / price_range) * (img_size[0] - 20) + 10)
        l_y = int((1 - (l - low_min) / price_range) * (img_size[0] - 20) + 10)
        c_y = int((1 - (c - low_min) / price_range) * (img_size[0] - 20) + 10)
        
        # Color based on direction
        color = (0, 255, 0) if c > o else (255, 0, 0)  # Green up, Red down
        
        # Draw high-low line
        cv2.line(img, (x, h_y), (x, l_y), (128, 128, 128), 1)
        
        # Draw open-close box
        top_y = min(o_y, c_y)
        bottom_y = max(o_y, c_y)
        if bottom_y - top_y < 2:
            bottom_y = top_y + 2
        
        cv2.rectangle(img, (x - candle_width//4, top_y), 
                     (x + candle_width//4, bottom_y), color, -1)
    
    return img

def candle_image_to_vector(df: pl.DataFrame, window_size: int = 20):
    """
    Convert candlestick patterns to feature vectors using CNN.
    Returns list of 512-dim vectors (zeros if GPU/CV unavailable).
    """
    if not CV_AVAILABLE:
        log.warning("Returning zero vectors - image dependencies not installed")
        return [np.zeros(512) for _ in range(len(df))]

    import torch  # Import here since we verified it's available

    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()

    vecs = []

    with torch.no_grad():
        for i in range(len(df)):
            # Get window of candles (current + previous)
            start_idx = max(0, i - window_size + 1)
            window = df.iloc[start_idx:i+1][["open", "high", "low", "close"]].values

            # Create pattern image
            img = create_candle_pattern_image(window)

            # Convert to tensor and extract features
            img_tensor = _tf(img).unsqueeze(0).to(_device)
            features = _model(img_tensor)

            # Flatten to 512-dim vector
            vec = features.squeeze().cpu().numpy()
            vecs.append(vec)

    return vecs
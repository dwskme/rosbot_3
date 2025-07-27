import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rosbot_lane_follower.unet_model import UNet
import torch

# Define the model path
model_path = r'd:\ros_ws3\src\rosbot_lane_follower\models\lane_unet.pth'

print("Loading model from:", model_path)

# Load the state dict first to inspect it
state_dict = torch.load(model_path, map_location='cpu', weights_only=False)  # Changed to False to handle potential custom objects

print("\n=== MODEL CHECKPOINT INFO ===")
print("Keys in saved model:")
for key in state_dict.keys():
    if hasattr(state_dict[key], 'shape'):
        print(f"  {key}: {state_dict[key].shape}")
    else:
        print(f"  {key}: {type(state_dict[key])}")

print("\n=== CURRENT MODEL INFO ===")
model = UNet()
print("Current model architecture:")
for name, param in model.named_parameters():
    print(f"  {name}: {param.shape}")

print("\n=== LOADING MODEL ===")
try:
    # Try loading with strict=False to see what happens
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    
    print("✓ Model loaded with strict=False")
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)
    
    if missing or unexpected:
        print("\n⚠️  There are mismatched keys. The model may not work correctly.")
    else:
        print("\n✓ Model loaded successfully with no issues!")
        
except Exception as e:
    print(f"✗ Error loading model: {e}")
    
    print("\n=== TROUBLESHOOTING ===")
    print("The saved model appears to have different architecture than your current UNet.")
    print("Possible solutions:")
    print("1. Check if your UNet model definition matches the one used during training")
    print("2. Look for different UNet constructors (e.g., UNet(n_channels=3, n_classes=1))")
    print("3. The saved model might be for a different version of your UNet class")
    
    # Try to suggest potential fixes based on the error
    print("\nBased on the error, it seems like:")
    print("- Saved model expects 384 input channels in dec2 layer, current model has 256")
    print("- Saved model expects 192 input channels in dec1 layer, current model has 128")
    print("This suggests the encoder part of your UNet might be different.")
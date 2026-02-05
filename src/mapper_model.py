
import torch
import torch.nn as nn

class BlindPainterMapper(nn.Module):
    def __init__(self, input_dim=512, output_dim=1024, hidden_dim=1024):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        x = x / x.norm(dim=-1, keepdim=True)
        return self.net(x)

if __name__ == "__main__":
    model = BlindPainterMapper()
    dummy_input = torch.randn(2, 512)
    output = model(dummy_input)
    print(f"Input: {dummy_input.shape} -> Output: {output.shape}")

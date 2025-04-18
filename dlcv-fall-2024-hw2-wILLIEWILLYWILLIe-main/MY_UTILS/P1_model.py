import torch 
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels = None, residual = False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size = 3, padding = 1, bias = False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size = 3, padding = 1, bias = False),
            nn.GroupNorm(1, out_channels)
        )
    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim = 256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), 
            DoubleConv(in_channels, in_channels, residual = True),
            DoubleConv(in_channels, out_channels),
        )
        self.emb_layer = nn.Sequential(
             nn.SiLU(),
             nn.Linear(emb_dim, out_channels),
        )
    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:,:,None, None].repeat(1,1,x.shape[-2], x.shape[-1])
        return x + emb

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dims = 256):
        super().__init__()

        self.up = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual = True),
            DoubleConv(in_channels, out_channels, in_channels//2)
        )
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dims, out_channels),
        )
    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:,:,None,None].repeat(1,1,x.shape[-2], x.shape[-1])
        return x+emb

class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first = True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    # def forward(self, x):
    #     x = x.view(-1, self.channels, self.size * self.size).swapaxes(1,2)
    #     x_ln = self.ln(x)
    #     attention_value, _ = self.mha(x_ln, x_ln, x_ln),
    #     attention_value = attention_value + x
    #     attention_value = self.ff_self(attention_value) + attention_value
    #     return attention_value.swapaxes(2,1).view(-1, self.channels, self.size, self.size)
    def forward(self, x):
        # Get the current spatial dimensions dynamically
        batch_size, channels, height, width = x.shape
        
        # Ensure the correct number of elements
        x = x.view(batch_size, self.channels, height * width).swapaxes(1, 2)

        # Layer normalization and multi-head attention
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        
        # Residual connection and feed-forward block
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        
        # Reshape back to original dimensions
        return attention_value.swapaxes(2, 1).view(batch_size, self.channels, height, width)

class UNet_conditional(nn.Module):
    def __init__(self, c_in = 3, c_out = 3, time_dim = 256, num_classes = None, device = "cuda"):
        super().__init__()
        self.device     = device
        self.time_dim   = time_dim

        self.inc    = DoubleConv(c_in, 64)
        self.down1  = DownBlock(64, 128)
        self.sa1    = SelfAttention(128, 32)
        self.down2  = DownBlock(128, 256)
        self.sa2    = SelfAttention(256, 16)
        self.down3  = DownBlock(256, 256)
        self.sa3    = SelfAttention(256, 8)

        self.bot1   = DoubleConv(256, 512)
        self.bot2   = DoubleConv(512,512)
        self.bot3   = DoubleConv(512,256)

        self.up1    = UpBlock(512,128)
        self.sa4    = SelfAttention(128,16)
        self.up2    = UpBlock(256,64)
        self.sa5    = SelfAttention(64,32)
        self.up3    = UpBlock(128,64)
        self.sa6    = SelfAttention(64,64)
        self.outc   = nn.Conv2d(64, c_out,kernel_size = 1)

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)
    
    def pos_encoding(self, t, channels):
        inv_freq = 1.0/(
            10000**(torch.arange(0 , channels, 2 , device = self.device).float() / channels)
        )
        pos_enc_a   = torch.sin(t.repeat(1, channels//2) * inv_freq)
        pos_enc_b   = torch.cos(t.repeat(1, channels//2) * inv_freq)
        pos_enc     = torch.cat([pos_enc_a, pos_enc_b], dim = -1)
        return pos_enc
    
    def forward(self, x, t, y=None):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        if y is not None:
            t += self.label_emb(y)

        # print(f"Input shape: {x.shape}")

        x1 = self.inc(x)
        # print(f"After inc (x1): {x1.shape}")

        x2 = self.down1(x1, t)
        # print(f"After down1 (x2): {x2.shape}")
        
        x2 = self.sa1(x2)
        # print(f"After sa1 (x2): {x2.shape}")

        x3 = self.down2(x2, t)
        # print(f"After down2 (x3): {x3.shape}")
        
        x3 = self.sa2(x3)
        # print(f"After sa2 (x3): {x3.shape}")

        x4 = self.down3(x3, t)
        # print(f"After down3 (x4): {x4.shape}")
        
        x4 = self.sa3(x4)
        # print(f"After sa3 (x4): {x4.shape}")

        x4 = self.bot1(x4)
        # print(f"After bot1: {x4.shape}")
        
        x4 = self.bot2(x4)
        # print(f"After bot2: {x4.shape}")
        
        x4 = self.bot3(x4)
        # print(f"After bot3: {x4.shape}")

        x = self.up1(x4, x3, t)
        # print(f"After up1: {x.shape}")
        
        x = self.sa4(x)
        # print(f"After sa4: {x.shape}")

        x = self.up2(x, x2, t)
        # print(f"After up2: {x.shape}")
        
        x = self.sa5(x)
        # print(f"After sa5: {x.shape}")

        x = self.up3(x, x1, t)
        # print(f"After up3: {x.shape}")
        
        x = self.sa6(x)  
        # print(f"After sa6: {x.shape}")

        output = self.outc(x)
        # print(f"Final output shape: {output.shape}")
        
        return output

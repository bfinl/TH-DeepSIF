from torch import nn
import torch
from skimage.filters import threshold_otsu
import numpy as np
import math


class EEGToBrainRegionTransformer(nn.Module):
    def __init__(self, input_channels=75, output_channels=994, d_model=256, nhead=8, num_layers=4, dim_feedforward=2048, dropout=0.1):
        super(EEGToBrainRegionTransformer, self).__init__()
        
        # Embedding layers to map input channels to d_model and output channels
        self.input_embedding = nn.Linear(input_channels, d_model)
        self.output_embedding = nn.Linear(d_model, output_channels)
        
        # Transformer Encoder-Decoder
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        
    def generate_positional_encoding(self, seq_len, d_model):
        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_encoding = torch.zeros(seq_len, d_model)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding.unsqueeze(1).to(next(self.parameters()).device)  # (seq_len, 1, d_model)
    
    def forward(self, src,src_key_padding_mask=None):
        # src shape: (seq_len, batch_size, input_channels)
        seq_len = src.shape[0]
        
        # Embed the input
        src = self.input_embedding(src)  # (seq_len, batch_size, d_model)
        
        # Generate positional encoding dynamically based on sequence length
        pos_encoding = self.generate_positional_encoding(seq_len, src.shape[2])
        src = src + pos_encoding  # (seq_len, batch_size, d_model)
        
        # Using src as both encoder input and decoder input
        transformer_output = self.transformer(src, src, src_key_padding_mask=src_key_padding_mask)
        
        # Project transformer output to output channels
        output = self.output_embedding(transformer_output)  # (seq_len, batch_size, output_channels)
        
        # Transpose back to (batch_size, seq_len, output_channels) for consistency
        output = output.transpose(0, 1)
        
        return output
    
    
    
    def count_parameters(self):
        # Calculate the number of trainable parameters
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class EnhancedEEGToBrainRegionTransformer(nn.Module):
    def __init__(self, input_channels=75, output_channels=994, d_model=512, nhead=8, num_layers=6,
                 dim_feedforward=4096, dropout=0.2, max_seq_len=500):
        super(EnhancedEEGToBrainRegionTransformer, self).__init__()

        # Convolutional embedding for local feature extraction
        self.conv1d = nn.Conv1d(in_channels=input_channels, out_channels=d_model, kernel_size=3, stride=1, padding=1)
        self.residual_conv = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, stride=1, padding=1)

        # Learnable positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(max_seq_len, d_model))

        # Transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, 
                nhead=nhead, 
                dim_feedforward=dim_feedforward, 
                dropout=dropout
            ),
            num_layers=num_layers
        )

        # Fully connected layer for mapping to output brain regions
        self.output_projection = nn.Linear(d_model, output_channels)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Activation and normalization layers
        self.activation = nn.ReLU()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src,src_key_padding_mask=None):
        """
        Forward pass for EEG to Brain Region Transformer.

        :param src: Input tensor of shape (batch_size, seq_len, input_channels)
        :return: Output tensor of shape (batch_size, seq_len, output_channels)
        """
        seq_len, batch_size, _ = src.shape

        # Apply convolutional embedding
        src = src.permute(1, 2, 0)  # Change shape to (batch_size, input_channels, seq_len)
        src = self.conv1d(src)  # (batch_size, d_model, seq_len)
        residual = self.residual_conv(src)  # Residual connection
        src = self.activation(self.norm((src + residual).permute(0,2,1)))  # Add residual and normalize
        src = src.permute(1, 0, 2)  # Change shape to (seq_len, batch_size, d_model)

        # Add positional encoding
        src = src + self.positional_encoding[:seq_len, :].unsqueeze(1)

        # Pass through transformer encoder
        encoded = self.transformer_encoder(src,src_key_padding_mask=src_key_padding_mask)  # (seq_len, batch_size, d_model)

        # Project to output space
        output = self.output_projection(encoded)  # (seq_len, batch_size, output_channels)
        output = self.dropout(output)

        # Transpose to (batch_size, seq_len, output_channels) for consistency
        output = output.permute(1, 0, 2)

        return output

    def count_parameters(self):
        """
        Count the number of trainable parameters in the model.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
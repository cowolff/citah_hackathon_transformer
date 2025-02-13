import torch
import torch.nn as nn
import torch.nn.functional as F

class VaeWrapper(nn.Module):
    def __init__(self, autoencoder):
        super(VaeWrapper, self).__init__()
        self.autoencoder = autoencoder
        self.latent_dim = autoencoder.latent_dim

    def forward(self, x):
        # Embedds the input for the transformer model
        x = self.autoencoder.encode(x)[0]
        return x

class DualProjectionTransformer(nn.Module):
    def __init__(self, image_vae, floorplan_vae, embed_dim, num_heads, num_layers, num_classes):
        super(DualProjectionTransformer, self).__init__()
        
        self.image_vae = VaeWrapper(image_vae)
        self.floorplan_vae = VaeWrapper(floorplan_vae)

        self.image_vae.requires_grad_(False)
        self.floorplan_vae.requires_grad_(False)

        self.image_proj = nn.Linear(self.image_vae.latent_dim, embed_dim)
        self.floorplan_proj = nn.Linear(self.floorplan_vae.latent_dim, embed_dim)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=2 * embed_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Linear(2 * embed_dim, num_classes)
        
    def forward(self, floorplan, image):
        """
        x: Tensor of shape (batch_size, seq_len, input_dim)
        """
        
        original_shape = floorplan.size()

        # Reshape the input from (batch_size, seq_len, input_dim) to (seq_len * batch_size, input_dim)
        floorplan = floorplan.view(original_shape[0] * original_shape[1], -1)
        image = image.view(original_shape[0] * original_shape[1], -1)

        floorplan_embeddings = self.floorplan_vae(floorplan)
        image_embeddings = self.image_vae(image)

        # Reshape the embeddings back to (seq_len, batch_size, embed_dim)
        floorplan_embeddings = floorplan_embeddings.view(original_shape[0], original_shape[1], -1)
        image_embeddings = image_embeddings.view(original_shape[0], original_shape[1], -1)
        
        floorplan_proj = self.floorplan_proj(floorplan_embeddings)
        image_proj = self.image_proj(image_embeddings)
        
        # Concatenation along the feature dimension
        combined = torch.cat([floorplan_proj, image_proj], dim=-1)  # (batch_size, seq_len, 2 * embed_dim)
        
        # Transformer expects (seq_len, batch_size, feature_dim)
        combined = combined.permute(1, 0, 2)
        
        # Pass through Transformer Encoder
        encoded = self.transformer_encoder(combined)  # (seq_len, batch_size, 2 * embed_dim)
        
        # Extract the classification token (we take the first token by convention)
        cls_token = encoded[0]  # (batch_size, 2 * embed_dim)
        
        # Pass through classifier
        out = self.classifier(cls_token)  # (batch_size, num_classes)
        return out
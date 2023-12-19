import torch
from torch import nn, optim
import lightning as L
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from typing import List

class CLIPVisionClassifier(L.LightningModule):
    def __init__(self, hidden_sizes:List):
            super().__init__()

            # Turn any grayscale images into 3 channels for CLIP image processor
            def convert(img):
                if img.shape[1] != 3:
                    return torch.broadcast_to(img, (1, 3, img.shape[2], img.shape[3]))
                return img
            self.enforce_3_channel_img = convert

            # Image processor for image encoder
            self.processor = CLIPImageProcessor.from_pretrained('openai/clip-vit-base-patch32', do_rescale=False)

            # Image encoder used by CLIP
            self.encoder = CLIPVisionModelWithProjection.from_pretrained('openai/clip-vit-base-patch32')
            # Freeze model
            for param in self.encoder.parameters():
                param.requires_grad = False

            # Lightweight neural net for embeddings
            hidden_sizes.insert(0, 512)
            hidden_sizes.append(101)
            layers = tuple([nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]) for i in range(len(hidden_sizes) - 1)])
            # Use GPU with neural net 
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.classifier = nn.Sequential(*layers).to(device)

            self.loss = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = self.enforce_3_channel_img(x)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = self.processor(images=x, return_tensors='pt')
        inputs.to(device)
        embeddings = self.encoder(**inputs).image_embeds
        y_hat = self.classifier(embeddings)
        loss = self.loss(y_hat, y)
        return loss

    def forward(self, x):
        x = self.enforce_3_channel_img(x)
        inputs = self.processor(images=x, return_tensors='pt')
        embeddings = self.encoder(**inputs).image_embeds
        y = self.classifier(embeddings)
        probs = nn.functional.softmax(y)
        return probs

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
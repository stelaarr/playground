import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from CLIP.clip import clip
from torch.utils.data import Dataset
from PIL import Image

### LOSS FUNCTIONS:
class GenGaussLoss(nn.Module):
    def __init__(self, reduction='mean', alpha_eps=1e-4, beta_eps=1e-4, resi_min=1e-4, resi_max=1e3) -> None:
        super(GenGaussLoss, self).__init__()
        self.reduction = reduction
        self.alpha_eps = alpha_eps
        self.beta_eps = beta_eps
        self.resi_min = resi_min
        self.resi_max = resi_max

    def forward(self, mean: torch.Tensor, one_over_alpha: torch.Tensor, beta: torch.Tensor, target: torch.Tensor):
        one_over_alpha1 = one_over_alpha + self.alpha_eps
        beta1 = beta + self.beta_eps

        resi = torch.abs(mean - target)
        resi = (resi * one_over_alpha1 * beta1).clamp(min=self.resi_min, max=self.resi_max)

        log_one_over_alpha = torch.log(one_over_alpha1)
        log_beta = torch.log(beta1)
        lgamma_beta = torch.lgamma(torch.pow(beta1, -1))

        l = resi - log_one_over_alpha + lgamma_beta - log_beta

        if self.reduction == 'mean':
            return l.mean()
        elif self.reduction == 'sum':
            return l.sum()
        else:
            print('Reduction not supported')
            return None


class TempCombLoss(nn.Module):
    def __init__(self, reduction='mean', alpha_eps=1e-4, beta_eps=1e-4, resi_min=1e-4, resi_max=1e3) -> None:
        super(TempCombLoss, self).__init__()
        self.reduction = reduction
        self.L_GenGauss = GenGaussLoss(reduction=self.reduction, alpha_eps=alpha_eps, beta_eps=beta_eps, resi_min=resi_min, resi_max=resi_max)
        self.L_l1 = nn.L1Loss(reduction=self.reduction)

    def forward(self, mean: torch.Tensor, one_over_alpha: torch.Tensor, beta: torch.Tensor, target: torch.Tensor, T1: float, T2: float):
        # Ensure target is a one-hot encoded tensor
        if target.dim() == 2 and target.size(1) == 10:  # Check if it's one-hot encoded
            target_indices = target.argmax(dim=1)  # Get class indices from one-hot encoding
        else:
            target_indices = target  # Assuming target is already in class index format
        # Ensure that mean has shape [batch_size, num_classes] 
        # and target_indices has shape [batch_size]
        mean = mean.view(-1, 10)  # Ensure mean is [64, 10] 
        target_indices = target_indices.view(-1, 1)  # Ensure target_indices is [64, 1]

        # Use index_select to select values from mean based on target_indices
        selected_means = mean.gather(1, target_indices)  # shape [64, 1]
        selected_means = selected_means.squeeze(1)  # shape [64]

        l1 = self.L_l1(selected_means, target_indices.float())  # Match the shape for L1 loss
        l2 = self.L_GenGauss(mean, one_over_alpha, beta, target_indices) 
        
        #l1 = self.L_l1(mean, target_indices) # Assuming target is one-hot encoded
        #l2 = self.L_GenGauss(mean, one_over_alpha, beta, target_indices)# Assuming target is one-hot encoded
        return T1 * l1 + T2 * l2

### BAYESIAN MLP

class BayesCap_MLP(nn.Module):
    def __init__(self, inp_dim, out_dim, hid_dim=512, num_layers=1, p_drop=0):
        super(BayesCap_MLP, self).__init__()
        layers = []
        for layer in range(num_layers):
            if layer == 0:
                layers.append(nn.Linear(inp_dim, hid_dim))
                layers.append(nn.ReLU())
            elif layer == num_layers // 2:
                layers.append(nn.Linear(hid_dim, hid_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p=p_drop))
            elif layer == num_layers - 1:
                layers.append(nn.Linear(hid_dim, out_dim))

        self.mod = nn.Sequential(*layers)

        self.block_mu = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

        self.block_alpha = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
        )

        self.block_beta = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x_intr = self.mod(x)
        x_mu = self.block_mu(x_intr)
        x_1alpha = self.block_alpha(x_intr)
        x_beta = self.block_beta(x_intr)
        return x_mu, x_1alpha, x_beta
    
### DATA LOADER MNIST

class CustomMNIST(Dataset):
    def __init__(self, data_dir, clip_model=None, preprocess=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.data_dir = data_dir
        self.images = []
        self.labels = []
        self.clip_model = clip_model
        self.device = device
        self.preprocess = preprocess

        # Load images and labels from the directory
        for label in os.listdir(data_dir):
            label_dir = os.path.join(data_dir, label)
            if os.path.isdir(label_dir):
                for img_file in os.listdir(label_dir):
                    img_path = os.path.join(label_dir, img_file)
                    self.images.append(img_path)
                    self.labels.append(int(label[-1]))  # Extract label from folder name

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')  # Convert to RGB

        image = self.preprocess(image)


        # Get CLIP features
        with torch.no_grad():
            image_features  = self.clip_model.encode_image(image.unsqueeze(0).to(self.device)) # Get CLIP features

        label = self.labels[idx]
        return image_features, label  # Return features instead of raw image
    


### TRAIN AND EVALUATE
def train_ProbVLM(model, train_loader, device='cuda', num_epochs=10, lr=1e-3):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = TempCombLoss()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        with tqdm(train_loader, unit='batch') as tepoch:
            # Changed images, labels to features, labels
            for features, labels in tepoch:  # <-- Change here
                features, labels = features.to(device), labels.to(device)  # Use features instead of images
                features = features.to(torch.float32)
                optimizer.zero_grad()
                
                # Pass features to the model
                mu, one_over_alpha, beta = model(features)  # <-- Change here
                # ** NEW LINE: Convert labels to one-hot encoding **
                labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=10).float()  
                
                # ** MODIFIED LINE: Use the modified custom loss with one-hot encoded labels **
                loss = criterion(mu, one_over_alpha, beta, labels_one_hot, 1.0, 1.0)
                #loss = criterion(mu, one_over_alpha, beta, labels.float(), 1.0, 1.0)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                tepoch.set_postfix(loss=running_loss / (tepoch.n + 1))

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')


def eval_ProbVLM(model, test_loader, device='cuda'):
    model.to(device)
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        # Changed images, labels to features, labels
        for features, labels in test_loader:  # <-- Change here
            features, labels = features.to(device), labels.to(device)  # Use features instead of images
            features = features.to(torch.float32)
            mu, one_over_alpha, beta = model(features)  # <-- Change here
            loss = TempCombLoss()(mu, one_over_alpha, beta, labels.float(), 1.0, 1.0)
            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    print(f'Average Loss: {avg_loss:.4f}')


#### INITIALIZE PROJECT
if __name__ == "__main__":
    
    # Load the CLIP model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clip_model, preprocess = clip.load("RN50", device=device, jit=False)  # Load the CLIP model
    
    # MNIST dataset
    # CLIP Preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),    # Resize the image to match CLIP's input size
        transforms.Grayscale(num_output_channels=3),  # Convert 1-channel grayscale to 3-channel
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                             std=[0.26862954, 0.26130258, 0.27577711])  # CLIP's normalization
    ])
    

    train_dataset = CustomMNIST(data_dir='./data/train', clip_model=clip_model,preprocess=preprocess, device=device)
    test_dataset = CustomMNIST(data_dir='./data/test',  clip_model=clip_model,preprocess=preprocess, device=device)

    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    # Model
    input_size = 1024  # Adjust to match CLIP feature size
    hidden_size = 512
    output_size = 10  # 10 classes for digits 0-9
    model = BayesCap_MLP(inp_dim=input_size, out_dim=output_size, hid_dim=hidden_size, num_layers=3, p_drop=0.3)


    # Training
    train_ProbVLM(model, train_loader, num_epochs=10)

    # Evaluation
    eval_ProbVLM(model, test_loader)
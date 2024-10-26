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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch import Tensor
from transformers import CLIPProcessor, CLIPModel

### Training and Evaluation utils

def emb_mae(x1, x2):
    m = torch.abs(x1-x2).mean()
    return m

def emb_mse(x1, x2):
    m = torch.pow(torch.abs(x1-x2),2).mean()
    return m

#### Loss

class GenGaussLoss(nn.Module):
	def __init__(
		self, reduction='mean',
		alpha_eps = 1e-4, beta_eps=1e-4,
		resi_min = 1e-4, resi_max=1e3
	) -> None:
		super(GenGaussLoss, self).__init__()
		self.reduction = reduction
		self.alpha_eps = alpha_eps
		self.beta_eps = beta_eps
		self.resi_min = resi_min
		self.resi_max = resi_max
	
	def forward(
		self, 
		mean: Tensor, one_over_alpha: Tensor, beta: Tensor, target: Tensor
	):
		one_over_alpha1 = one_over_alpha + self.alpha_eps
		beta1 = beta + self.beta_eps

		resi = torch.abs(mean - target)
		# resi = torch.pow(resi*one_over_alpha1, beta1).clamp(min=self.resi_min, max=self.resi_max)
		resi = (resi*one_over_alpha1*beta1).clamp(min=self.resi_min, max=self.resi_max)
		## check if resi has nans
		if torch.sum(resi != resi) > 0:
			print('resi has nans!!')
			return None
		
		log_one_over_alpha = torch.log(one_over_alpha1)
		log_beta = torch.log(beta1)
		lgamma_beta = torch.lgamma(torch.pow(beta1, -1))
		
		if torch.sum(log_one_over_alpha != log_one_over_alpha) > 0:
			print('log_one_over_alpha has nan')
		if torch.sum(lgamma_beta != lgamma_beta) > 0:
			print('lgamma_beta has nan')
		if torch.sum(log_beta != log_beta) > 0:
			print('log_beta has nan')
		
		l = resi - log_one_over_alpha + lgamma_beta - log_beta

		if self.reduction == 'mean':
			return l.mean()
		elif self.reduction == 'sum':
			return l.sum()
		else:
			print('Reduction not supported')
			return None

class TempCombLoss(nn.Module):
	def __init__(
		self, reduction='mean',
		alpha_eps = 1e-4, beta_eps=1e-4,
		resi_min = 1e-4, resi_max=1e3
	) -> None:
		super(TempCombLoss, self).__init__()
		self.reduction = reduction
		self.alpha_eps = alpha_eps
		self.beta_eps = beta_eps
		self.resi_min = resi_min
		self.resi_max = resi_max

		self.L_GenGauss = GenGaussLoss(
			reduction=self.reduction,
			alpha_eps=self.alpha_eps, beta_eps=self.beta_eps, 
			resi_min=self.resi_min, resi_max=self.resi_max
		)
		self.L_l1 = nn.L1Loss(reduction=self.reduction)
	
	def forward(
		self,
		mean: Tensor, one_over_alpha: Tensor, beta: Tensor, target: Tensor,
		T1: float, T2: float
	):
		l1 = self.L_l1(mean, target)
		l2 = self.L_GenGauss(mean, one_over_alpha, beta, target)
		l = T1*l1 + T2*l2

		return l

"""
##### PREPROCESS
def preprocess_mnist():
    # Using torchvision transforms instead of CLIP preprocess
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to CLIP input size
        transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3-channel RGB
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize(  # Normalize based on CLIP's expectations
            mean=(0.48145466, 0.4578275, 0.40821073), 
            std=(0.26862954, 0.26130258, 0.27577711)
        ),
    ])
    return transform


class CustomMNIST(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        # Convert label to desired format
        formatted_label = f'digit {label}'
        return image, formatted_label
"""


def load_model(device, model_path=None):
    # load zero-shot CLIP model
    model, _ = clip.load(name='ViT-B/32',
                         device=device, jit=False,
                         loss_type='contrastive')
    if model_path is None:
        # Convert the dtype of parameters from float16 to float32
        for name, param in model.named_parameters():
            param.data = param.data.type(torch.float32)
    else:
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['state_dict'])
        for name, param in model.named_parameters():
            param.data = param.data.type(torch.float32)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    return model

####  NETWORKS

class BayesCap_MLP(nn.Module):
    '''
    Baseclass to create a simple MLP
    Inputs
        inp_dim: int, Input dimension
        out_dim: int, Output dimension
        hid_dim: int, hidden dimension
        num_layers: Number of hidden layers
        p_drop: dropout probability 
    '''
    def __init__(
        self, 
        inp_dim, 
        out_dim,
        hid_dim=512, 
        num_layers=1, 
        p_drop=0,
    ):
        super(BayesCap_MLP, self).__init__()
        mod = []
        for layer in range(num_layers):
            print(f'Layer {layer}', flush=True)
            if layer==0:
                incoming = inp_dim
                outgoing = hid_dim
                mod.append(nn.Linear(incoming, outgoing))
                mod.append(nn.ReLU())
                print('hidden layers',outgoing, flush=True)
            elif layer==num_layers//2:
                incoming = hid_dim
                outgoing = hid_dim
                mod.append(nn.Linear(incoming, outgoing))
                mod.append(nn.ReLU())
                mod.append(nn.Dropout(p=p_drop))
                print('hidden layers',outgoing, flush=True)
            elif layer==num_layers-1:
                incoming = hid_dim
                outgoing = out_dim
                mod.append(nn.Linear(incoming, outgoing))
        self.mod = nn.Sequential(*mod)
        print('architecture',self.mod,flush=True)

        self.block_mu = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

        self.block_alpha = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            # nn.Linear(out_dim, out_dim),
            # nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
        )

        self.block_beta = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            # nn.Linear(out_dim, out_dim),
            # nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
        )
    
    def forward(self, x):
        print('dbg1', x.shape, flush=True)   
        x_intr = self.mod(x)
        print('dbg', x_intr.shape, x.shape, flush=True)
        x_intr = x_intr + x
        x_mu = self.block_mu(x_intr)
        x_1alpha = self.block_alpha(x_intr)
        x_beta = self.block_beta(x_intr)
        return x_mu, x_1alpha, x_beta
    
    
    
    
class BayesCLIP(nn.Module):
    def __init__(
        self,
        model_path=None,
        device='cuda',
    ):
        super(BayesCLIP, self).__init__()
        self.clip_model = load_model(device, model_path)
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.img_BayesCap = BayesCap_MLP(inp_dim=512, out_dim=512, hid_dim=512, num_layers=3, p_drop=0.3).to(device)
        self.txt_BayesCap = BayesCap_MLP(inp_dim=512, out_dim=512, hid_dim=512, num_layers=3, p_drop=0.3).to(device)

    def forward(self, i_inputs, t_inputs):
        i_features, t_features = self.clip_model(i_inputs, t_inputs)

        img_mu, img_1alpha, img_beta = self.img_BayesCap(i_features)
        txt_mu, txt_1alpha, txt_beta = self.txt_BayesCap(t_features)

        return (img_mu, img_1alpha, img_beta), (txt_mu, txt_1alpha, txt_beta), (i_features, t_features)


class BayesCap_for_CLIP(nn.Module):
    def __init__(
        self,
        inp_dim=512,
        out_dim=512,
        hid_dim=256,
        num_layers=3,
        p_drop=0.1,
    ):
        super(BayesCap_for_CLIP, self).__init__()
        self.img_BayesCap = BayesCap_MLP(inp_dim=inp_dim, out_dim=out_dim, hid_dim=hid_dim, num_layers=num_layers, p_drop=p_drop)
        self.txt_BayesCap = BayesCap_MLP(inp_dim=inp_dim, out_dim=out_dim, hid_dim=hid_dim, num_layers=num_layers, p_drop=p_drop)

    def forward(self, i_features, t_features):
        
        print('dbg', i_features.shape, t_features.shape)
        img_mu, img_1alpha, img_beta = self.img_BayesCap(i_features)
        txt_mu, txt_1alpha, txt_beta = self.txt_BayesCap(t_features)

        return (img_mu, img_1alpha, img_beta), (txt_mu, txt_1alpha, txt_beta)

    
#### LOAD DATA


"""
def load_mnist_data(batch_size=64):
    # Load MNIST dataset using torchvision
    transform = preprocess_mnist()
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # Wrap datasets with the custom dataset class
    train_dataset = CustomMNIST(train_dataset)
    test_dataset = CustomMNIST(test_dataset)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
"""


# Update load_mnist_data to accept processor
def load_mnist_data(batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to CLIP input size
        transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3-channel RGB
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize(  # Normalize based on CLIP's expectations
            mean=(0.48145466, 0.4578275, 0.40821073), 
            std=(0.26862954, 0.26130258, 0.27577711)
        ),
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader




###### TRAIN
def train_ProbVLM(
    CLIP_Net,
    BayesCap_Net,
    clip_processor,  # Add this line
    train_loader,
    eval_loader,
    Cri = TempCombLoss(),
    device='cuda',
    dtype=torch.cuda.FloatTensor(),
    init_lr=1e-4,
    num_epochs=50,
    eval_every=1,
    ckpt_path='../ckpt',
    cross_modal_lambda=1e-4,
    T1=1e0,
    T2=5e-2
):
    CLIP_Net.to(device)
    CLIP_Net.eval()
    ##
    BayesCap_Net.to(device)
    BayesCap_Net.img_BayesCap.train()
    BayesCap_Net.txt_BayesCap.train()
    ##
    optimizer = torch.optim.Adam(
        list(BayesCap_Net.img_BayesCap.parameters())+list(BayesCap_Net.txt_BayesCap.parameters()), 
        lr=init_lr
    )
    optim_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    score = 1e8
    all_loss = []
    for eph in range(num_epochs):
        eph_loss = 0
        BayesCap_Net.train()
        with tqdm(train_loader, unit='batch') as tepoch:
            for (idx, batch) in enumerate(tepoch):
                if idx>2000:
                    break
                tepoch.set_description('Epoch {}'.format(eph))
                ##
                #xI, xT  = batch[0].to(device), batch[1].to(device)
                # The batch is already preprocessed
                images = [Image.fromarray((torch.clamp(item.cpu(), 0, 1).numpy() * 255).astype('uint8').transpose(1, 2, 0))
                    for item in batch[0]]
                labels = batch[1]  # Labels

                # Create the text labels
                texts = [f'digit {item}' for item in labels.tolist()]
                inputs = clip_processor(
                    text=texts,  # Assuming batch contains 'texts'
                    images=images,  # Assuming batch contains 'images'
                    return_tensors="pt",
                    padding=True
                ).to(device)
                
                xI = inputs['pixel_values']  # Image features
                xT = inputs['input_ids']
                print(f'size image before clip {xI.shape}, and size text before clip {xT.shape}', flush=True)
                # Get embeddings from CLIP model
               
                    
                with torch.no_grad():
                    xfI = CLIP_Net.get_image_features(pixel_values=xI)
                    xfT = CLIP_Net.get_text_features(input_ids=xT, attention_mask=inputs['attention_mask'])
                
                
                #with torch.no_grad():
                #xfI, xfT = CLIP_Net(xI, xT)
                # xI, xT = xI.type(dtype), xT.type(dtype)
                # pass them through the network
                # Ensure both tensors have the same dtype
                xfI = xfI.to(torch.float32)
                xfT = xfT.to(torch.float32)
                print(f'size image after clip {xfI.shape}, and size text after clip {xfT.shape}', flush=True)

                (img_mu, img_1alpha, img_beta), (txt_mu, txt_1alpha, txt_beta) = BayesCap_Net(xfI, xfT)
                
                optimizer.zero_grad()
                loss_i = Cri(img_mu, img_1alpha, img_beta, xfI, T1=T1, T2=T2)
                loss_t = Cri(txt_mu, txt_1alpha, txt_beta, xfT, T1=T1, T2=T2)
                #cross modal terms
                loss_i4t = Cri(img_mu, img_1alpha, img_beta, xfT, T1=T1, T2=T2)
                loss_t4i = Cri(txt_mu, txt_1alpha, txt_beta, xfI, T1=T1, T2=T2)
                loss = loss_i + loss_t + cross_modal_lambda*(loss_i4t + loss_t4i)
                # print(loss)
                loss.backward()
                optimizer.step()
                ##
                eph_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())
            eph_loss /= len(train_loader)
            all_loss.append(eph_loss)
            print('Avg. loss: {}'.format(eph_loss))
        # evaluate and save the models
        torch.save(BayesCap_Net.state_dict(), ckpt_path+'_last.pth')
        if eph%eval_every == 0:
            curr_score = eval_ProbVLM(
                CLIP_Net,
                BayesCap_Net, clip_processor,
                eval_loader,
                device=device,
                dtype=dtype,
            )
            print('current score: {} | Last best score: {}'.format(curr_score, score))
            if curr_score <= score:
                score = curr_score
                torch.save(BayesCap_Net.state_dict(), ckpt_path+'_best.pth')
    optim_scheduler.step()
    
    
def eval_ProbVLM(
    CLIP_Net,
    BayesCap_Net,
    clip_processor,
    eval_loader,
    device='cuda',
    dtype=torch.cuda.FloatTensor,
):
    CLIP_Net.to(device)
    CLIP_Net.eval()
    BayesCap_Net.to(device)
    BayesCap_Net.eval()

    mean_mse = 0
    mean_mae = 0
    num_imgs = 0
    list_error = []
    list_var = []
    with tqdm(eval_loader, unit='batch') as tepoch:
        for (idx, batch) in enumerate(tepoch):
            tepoch.set_description('Validating ...')
            ##
            #xI, xT  = batch[0].to(device), batch[1].to(device)
            # xI, xT = xI.type(dtype), xT.type(dtype)
            
            images = [Image.fromarray((torch.clamp(item.cpu(), 0, 1).numpy() * 255).astype('uint8').transpose(1, 2, 0))
                    for item in batch[0]]
            labels = batch[1] 
            texts = [f'digit {item}' for item in labels.tolist()]
            inputs = clip_processor(
                    text=texts,  # Assuming batch contains 'texts'
                    images=images,  # Assuming batch contains 'images'
                    return_tensors="pt",
                    padding=True
                ).to(device)

            xI = inputs['pixel_values']  # Image features
            xT = inputs['input_ids']
            
            with torch.no_grad():
                    xfI = CLIP_Net.get_image_features(pixel_values=xI)
                    xfT = CLIP_Net.get_text_features(input_ids=xT, attention_mask=inputs['attention_mask'])
                    xfI = xfI.to(torch.float32)
                    xfT = xfT.to(torch.float32)
                    (img_mu, img_1alpha, img_beta), (txt_mu, txt_1alpha, txt_beta) = BayesCap_Net(xfI, xfT)
            # pass them through the network
            
                
            n_batch = img_mu.shape[0]
            for j in range(n_batch):
                num_imgs += 1
                mean_mse += emb_mse(img_mu[j], xfI[j]) + emb_mse(txt_mu[j], xfT[j])
                mean_mae += emb_mae(img_mu[j], xfI[j]) + emb_mae(txt_mu[j], xfT[j])
            ##
        mean_mse /= num_imgs
        mean_mae /= num_imgs
        print(
            'Avg. MSE: {} | Avg. MAE: {}'.format
            (
                mean_mse, mean_mae 
            )
        )
    return mean_mae




def main():
   

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

    # Load the data
    train_loader, test_loader = load_mnist_data(batch_size=64)
    
    # Define the BayesCap_MLP model
    inp_dim = 512  # Assuming 512 dimensions from CLIP ViT-B/32 encoded features
    out_dim = 512  # Assuming we're outputting the same dimension
    bayescap_mlp = BayesCap_for_CLIP(inp_dim, out_dim)
    #print(type(bayescap_mlp), flush=True)  # This should output <class '__main__.BayesCLIP'> or <class '__main__.BayesCap_for_CLIP'>

    # Train the model
    train_ProbVLM(model, bayescap_mlp, clip_processor, train_loader, test_loader, device=device)

if __name__ == "__main__":
    main()
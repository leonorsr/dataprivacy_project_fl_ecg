"""Arquiteturas partilhadas com os smoke tests centralizados (sem executar notebooks)."""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
SIGNAL_LENGTH=6000
def temporal_mask(lengths, steps, input_steps=None):
    input_steps = SIGNAL_LENGTH if input_steps is None else input_steps
    valid = torch.ceil(lengths.float()*steps/input_steps).long().clamp(1, steps)
    return torch.arange(steps, device=lengths.device)[None, :] < valid[:, None]

def masked_mean(x, lengths):
    mask = temporal_mask(lengths, x.shape[-1]).to(x.dtype)[:, None, :]
    return (x*mask).sum(dim=-1)/mask.sum(dim=-1).clamp_min(1)

def conv_block(cin, cout, kernel=7, stride=1):
    return nn.Sequential(nn.Conv1d(cin, cout, kernel, stride=stride, padding=kernel//2, bias=False), nn.BatchNorm1d(cout), nn.ReLU())

class FusionHead(nn.Module):
    def __init__(self, signal_dim, feature_dim):
        super().__init__()
        self.tabular = nn.Sequential(nn.Linear(feature_dim, 16), nn.ReLU())
        self.classifier = nn.Sequential(nn.Dropout(.2), nn.Linear(signal_dim+16, 4))
    def forward(self, signal_embedding, features):
        return self.classifier(torch.cat([signal_embedding, self.tabular(features)], dim=1))

class CNNTrunk(nn.Sequential):
    def __init__(self):
        super().__init__(conv_block(1,16,15,4), conv_block(16,32,9,2), nn.MaxPool1d(2), conv_block(32,64,7,2), nn.MaxPool1d(2))

class AdditiveAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(channels, 32), nn.Tanh(), nn.Linear(32,1,bias=False))
    def forward(self, sequence, mask):
        weights = self.score(sequence).squeeze(-1).masked_fill(~mask, -torch.inf).softmax(dim=1)
        return (sequence*weights.unsqueeze(-1)).sum(dim=1)

class CBAM1D(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.shared_mlp = nn.Sequential(nn.Linear(channels,max(1,channels//reduction)),nn.ReLU(),nn.Linear(max(1,channels//reduction),channels))
        self.temporal = nn.Conv1d(2,1,7,padding=3,bias=False)
    def forward(self, x, mask):
        m = mask[:, None, :].to(x.dtype)
        avg = (x*m).sum(-1)/m.sum(-1).clamp_min(1)
        maximum = x.masked_fill(~mask[:,None,:], -torch.inf).amax(-1)
        channel = torch.sigmoid(self.shared_mlp(avg)+self.shared_mlp(maximum)).unsqueeze(-1)
        x = x*channel*m
        temporal = torch.sigmoid(self.temporal(torch.cat([x.mean(1,keepdim=True),x.amax(1,keepdim=True)],dim=1)))
        return x*temporal*m

class CNN1D(nn.Module):
    def __init__(self, feature_dim):
        super().__init__();self.trunk=CNNTrunk();self.head=FusionHead(64,feature_dim)
    def forward(self,x,features,lengths):
        return self.head(masked_mean(self.trunk(x),lengths),features)

class CNNBiLSTM(nn.Module):
    def __init__(self,feature_dim,attention=False,cbam=False):
        super().__init__();self.trunk=CNNTrunk();self.cbam=CBAM1D(64) if cbam else None
        self.lstm=nn.LSTM(64,32,num_layers=1,batch_first=True,bidirectional=True)
        self.attention=AdditiveAttention(64) if attention else None
        self.head=FusionHead(64,feature_dim)
    def forward(self,x,features,lengths):
        z=self.trunk(x);mask=temporal_mask(lengths,z.shape[-1])
        if self.cbam is not None:z=self.cbam(z,mask)
        packed=pack_padded_sequence(z.transpose(1,2),mask.sum(1).cpu(),batch_first=True,enforce_sorted=False)
        encoded,_=self.lstm(packed)
        sequence,_=pad_packed_sequence(encoded,batch_first=True,total_length=z.shape[-1])
        embedding=self.attention(sequence,mask) if self.attention is not None else (sequence*mask[:,:,None]).sum(1)/mask.sum(1,keepdim=True)
        return self.head(embedding,features)

class BasicBlock1D(nn.Module):
    def __init__(self,cin,cout,stride=1):
        super().__init__();self.conv1=conv_block(cin,cout,3,stride)
        self.conv2=nn.Sequential(nn.Conv1d(cout,cout,3,padding=1,bias=False),nn.BatchNorm1d(cout))
        self.skip=nn.Identity() if cin==cout and stride==1 else nn.Sequential(nn.Conv1d(cin,cout,1,stride=stride,bias=False),nn.BatchNorm1d(cout))
    def forward(self,x):return F.relu(self.conv2(self.conv1(x))+self.skip(x))

class ResNet34ECG(nn.Module):
    def __init__(self,feature_dim,width=8):
        super().__init__();self.stem=nn.Sequential(conv_block(1,width,15,2),nn.MaxPool1d(3,stride=2,padding=1))
        blocks=[];cin=width
        for stage,count in enumerate([3,4,6,3]):
            cout=width*(2**stage)
            for index in range(count):
                blocks.append(BasicBlock1D(cin,cout,stride=2 if stage>0 and index==0 else 1));cin=cout
        self.blocks=nn.Sequential(*blocks);self.head=FusionHead(cin,feature_dim)
    def forward(self,x,features,lengths):return self.head(masked_mean(self.blocks(self.stem(x)),lengths),features)

class InvertedResidual1D(nn.Module):
    def __init__(self,cin,cout,stride,expansion):
        super().__init__();hidden=int(cin*expansion);layers=[]
        if expansion!=1:layers.extend([nn.Conv1d(cin,hidden,1,bias=False),nn.BatchNorm1d(hidden),nn.ReLU6()])
        layers.extend([nn.Conv1d(hidden,hidden,3,stride=stride,padding=1,groups=hidden,bias=False),nn.BatchNorm1d(hidden),nn.ReLU6(),nn.Conv1d(hidden,cout,1,bias=False),nn.BatchNorm1d(cout)])
        self.body=nn.Sequential(*layers);self.residual=stride==1 and cin==cout
    def forward(self,x):return x+self.body(x) if self.residual else self.body(x)

class MobileNetV2ECG(nn.Module):
    def __init__(self,feature_dim,width_mult=.25):
        super().__init__()
        def channels(value):return max(8,int(round(value*width_mult/8))*8)
        cin=channels(32);layers=[nn.Conv1d(1,cin,3,stride=2,padding=1,bias=False),nn.BatchNorm1d(cin),nn.ReLU6()]
        self.stage_config=[(1,16,1,1),(6,24,2,2),(6,32,3,2),(6,64,4,2),(6,96,3,1),(6,160,3,2),(6,320,1,1)]
        for expansion,base,count,stride in self.stage_config:
            cout=channels(base)
            for index in range(count):
                layers.append(InvertedResidual1D(cin,cout,stride if index==0 else 1,expansion));cin=cout
        last=channels(1280);layers.extend([nn.Conv1d(cin,last,1,bias=False),nn.BatchNorm1d(last),nn.ReLU6()])
        self.trunk=nn.Sequential(*layers);self.head=FusionHead(last,feature_dim)
    def forward(self,x,features,lengths):return self.head(masked_mean(self.trunk(x),lengths),features)

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=nn.Sequential(conv_block(1,8,9,4),conv_block(8,16,9,4))
        self.decoder=nn.Sequential(nn.ConvTranspose1d(16,8,8,stride=4,padding=2),nn.ReLU(),nn.ConvTranspose1d(8,1,8,stride=4,padding=2))
    def forward(self,x):return self.decoder(self.encoder(x))[..., :x.shape[-1]]

class AutoencoderCNN(nn.Module):
    def __init__(self,feature_dim,encoder):
        super().__init__();self.encoder=copy.deepcopy(encoder)
        self.cnn=nn.Sequential(conv_block(16,32,5,2),conv_block(32,64,5,2));self.head=FusionHead(64,feature_dim)
    def forward(self,x,features,lengths):return self.head(masked_mean(self.cnn(self.encoder(x)),lengths),features)
    def cam_target(self):return self.cnn[-1][0]


class GRUECG(nn.Module):
    """GRU com fusão de features; pooling temporal fixo limita o custo do smoke test."""
    def __init__(self, feature_dim, layers, signal_length):
        super().__init__()
        self.hop = max(1, (signal_length + 499) // 500)
        self.gru = nn.GRU(1, 64, num_layers=layers, batch_first=True,
                          dropout=.2 if layers > 1 else 0.)
        self.dropout = nn.Dropout(.2)
        self.head = FusionHead(64, feature_dim)
    def forward(self, x, features, lengths):
        if self.hop > 1:
            x = F.avg_pool1d(x, self.hop, self.hop, ceil_mode=True)
        valid = ((lengths + self.hop - 1) // self.hop).clamp(1, x.shape[-1])
        packed = pack_padded_sequence(x.transpose(1, 2), valid.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        return self.head(self.dropout(hidden[-1]), features)

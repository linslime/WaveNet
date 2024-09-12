import glob
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset,DataLoader

import torchaudio
import librosa
from torchaudio import transforms
from model import WaveNet

import matplotlib.pyplot as plt
import IPython.display
import numpy as np
import random


class VCTK(Dataset):
	def __init__(self, path='./VCTK/', speaker='p225', transform=None, sr=16000, top_db=10):
		self.wav_list = glob.glob(path + speaker + '/*.wav')
		self.wav_ids = sorted([f.split('/')[-1] for f in glob.glob(path + '*')])
		self.transform = transform
		self.sr = sr
		self.top_db = top_db
	
	def __getitem__(self, index):
		f = self.wav_list[index]
		audio, _ = librosa.load(f, sr=self.sr, mono=True)
		audio, _ = librosa.effects.trim(audio, top_db=self.top_db, frame_length=2048)
		audio = np.clip(audio, -1, 1)
		wav_tensor = torch.from_numpy(audio).unsqueeze(1)
		wav_id = f.split('/')[3]
		if self.transform is not None:
			wav_tensor = self.transform(wav_tensor)
		
		return wav_tensor
	
	def __len__(self):
		return len(self.wav_list)
	
t = transforms.Compose([
        transforms.MuLawEncoding(),
        transforms.LC2CL()])

def collate_fn_(batch_data, max_len=40000):
    audio = batch_data[0]
    audio_len = audio.size(1)
    if audio_len > max_len:
        idx = random.randint(0,audio_len - max_len)
        return audio[:,idx:idx+max_len]
    else:
        return audio

vctk = VCTK(speaker='p225',transform=t,sr=16000)
training_data = DataLoader(vctk,batch_size=1, shuffle=True,collate_fn=collate_fn_)

model = WaveNet()
train_step = optim.Adam(model.parameters(),lr=2e-3, eps=1e-4)

scheduler = optim.lr_scheduler.MultiStepLR(train_step, milestones=[50,150,250], gamma=0.5)

for epoch in range(1000):
	loss_ = []
	scheduler.step()
	for data in training_data:
		data = Variable(data).cuda()
		x = data[:, :-1]
		
		logits = model(x)
		y = data[:, -logits.size(2):]
		loss = F.cross_entropy(logits.transpose(1, 2).contiguous().view(-1, 256), y.view(-1))
		train_step.zero_grad()
		loss.backward()
		train_step.step()
		loss_.append(loss.data[0])
	
	print(epoch, np.mean(loss_))
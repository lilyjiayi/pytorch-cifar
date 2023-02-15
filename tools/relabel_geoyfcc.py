#!/usr/bin/env python
# coding: utf-8

# In[60]:


import numpy as np
import pandas as pd
import mapply
import torch
import open_clip
from torchvision.datasets.folder import default_loader
from torch.utils.data import DataLoader
import wn
from tqdm import tqdm, trange
from IPython.display import Image, display
import matplotlib.pyplot as plt


# In[2]:


mapply.init(n_workers=20)
_ = torch.set_grad_enabled(False)
torch.multiprocessing.set_sharing_strategy('file_system')


# ## Load clip

# In[3]:


model, _, test_transform = open_clip.create_model_and_transforms('ViT-bigG-14', pretrained='laion2b_s39b_b160k')
# model, _, test_transform = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')


# In[4]:


model = model.cuda().eval()


# ## Load geo-yfcc dataset, compute image features

# In[45]:


geoyfcc_metadata = pd.read_pickle('/juice2/u/nlp/data/geo_yfcc/one_label.pkl')


# In[46]:


class GeoYfccDataset:
    def __init__(self, df, transform):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        filename, target = self.df.iloc[index]['file_name'], self.df.iloc[index]['label_ids'][0]
        img_path = '/juice2/u/nlp/data/geo_yfcc/data/' + filename[:3] + '/' + filename + '.jpg'
        return self.transform(default_loader(img_path)), target


# In[47]:


loader = DataLoader(GeoYfccDataset(geoyfcc_metadata, test_transform), batch_size=100, num_workers=20)


# In[7]:


features, labels = [], []
for images, targets in tqdm(loader):
    image_features = model.encode_image(images.cuda())
    image_features /= image_features.norm(dim=-1, keepdim=True)
    features.append(image_features)
    labels.append(targets)
features, labels = torch.cat(features, dim=0), torch.cat(labels, dim=0)


# In[8]:


torch.save((features, labels), 'features.pt')


# ## Load lemma information for all imagenet-21k classes, compute text features

# In[7]:


imagenet21k_metadata = pd.read_csv('/juice2/u/nlp/data/geo_yfcc/imagenet21k_wordnet_ids.txt', sep='\t', header=None, names=['wnid'])
en = wn.Wordnet('omw-en')


# In[8]:


imagenet21k_metadata['lemmas'] = imagenet21k_metadata.mapply(lambda row: en.synset(f'omw-en-{row.wnid[1:]}-n').lemmas(), axis='columns')
imagenet21k_metadata['definition'] = imagenet21k_metadata.mapply(lambda row: en.synset(f'omw-en-{row.wnid[1:]}-n').definition(), axis='columns')


# In[9]:


tokenizer = open_clip.get_tokenizer('ViT-bigG-14')


# In[10]:


text_inputs = tokenizer([', '.join(lemmas) for lemmas in imagenet21k_metadata.lemmas]).cuda()


# In[120]:


text_features = []
for i in trange(0, len(text_inputs), 100):
    text_chunk = text_inputs[i:i+100]
    text_feature = model.encode_text(text_chunk)
    text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
    text_features.append(text_feature)
text_features = torch.cat(text_features, dim=0)


# In[121]:


torch.save(text_features, 'text_features.pt')


# ## Compute model predictions

# In[122]:


features, labels = torch.load('features.pt')
text_features = torch.load('text_features.pt')


# In[123]:


preds = []
for i in trange(0, len(features), 10_000):
    logits = features[i:i+10_000] @ text_features.T
    pred = logits.argmax(dim=1)
    preds.append(pred)
preds = torch.cat(preds, dim=0)


# ## Look at predictions for randomly sampled images

# In[124]:


for i in np.random.choice(len(geoyfcc_metadata), size=10, replace=False):
    print('clip predicted imagenet-21k class:', imagenet21k_metadata.iloc[preds[i].item()].lemmas)
    filename = geoyfcc_metadata.iloc[i]['file_name']
    img_path = '/juice2/u/nlp/data/geo_yfcc/data/' + filename[:3] + '/' + filename + '.jpg'
    display(Image(filename=img_path))


# ## (perhaps not needed) for a given original geo_yfcc label, output images and clip predictions

# In[127]:


orig_label = 4
orig_label_indices = torch.where(labels == orig_label)[0]
shuffle_indices = torch.randperm(len(orig_label_indices))[:10]
for i in orig_label_indices[shuffle_indices]:
    print('clip predicted imagenet-21k class:', imagenet21k_metadata.iloc[preds[i.item()].item()].lemmas)
    filename = geoyfcc_metadata.iloc[i.item()]['file_name']
    img_path = '/juice2/u/nlp/data/geo_yfcc/data/' + filename[:3] + '/' + filename + '.jpg'
    display(Image(filename=img_path))


# ## Print most populous predicted classes

# In[128]:


pred_idx, pred_counts = preds.unique(return_counts=True)
pred_counts_argsort = pred_counts.argsort(descending=True)
pred_idx, pred_counts = pred_idx[pred_counts_argsort], pred_counts[pred_counts_argsort]


# In[138]:


for i in range(200):
    print(f'clip predicted {pred_counts[i]} counts of class index {pred_idx[i].item()}: {imagenet21k_metadata.iloc[pred_idx[i].item()].lemmas}')


# ## Look at predictions by predicted class

# In[136]:


pred_label = 7314
print('images that clip predicted for imagenet-21k class:', imagenet21k_metadata.iloc[pred_label].lemmas)
pred_label_indices = torch.where(preds == pred_label)[0]
shuffle_indices = torch.randperm(len(pred_label_indices))[:10]
for i in pred_label_indices[shuffle_indices]:
    filename = geoyfcc_metadata.iloc[i.item()]['file_name']
    img_path = '/juice2/u/nlp/data/geo_yfcc/data/' + filename[:3] + '/' + filename + '.jpg'
    display(Image(filename=img_path))


# In[ ]:





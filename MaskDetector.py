#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install -Uqq fastai')
get_ipython().system('pip install -Uqq timm')


# In[24]:


get_ipython().system('pip install -Uqq Pathlib')


# In[50]:


get_ipython().system('pip install -Uqq voila')


# In[51]:


get_ipython().system('pip install -Uqq opencv-python')


# In[25]:


from fastai.imports import *
from fastai.vision.all import *
import timm


# In[26]:


def getimageloader(datasetpath = Path('./dataset')):
  path = datasetpath
  files = get_image_files(path)
  dls = ImageDataLoaders.from_path_func(path, files, parent_label, valid_pct=0.2,item_tfms=Resize(229), batch_tfms=aug_transforms(size=224))
  return dls


# In[27]:


def get_model():
  if Path('./maskmodel.pkl').exists():
    print('Found existing model... Loading...')
    learn = load_learner('maskmodel.pkl')
    return learn
  learn = vision_learner(dls, 'inception_v3', metrics=accuracy)
  if Path('/content/drive/MyDrive/MaskDetector/dataset/models/model.pth').exists():
    learn.load('/content/drive/MyDrive/MaskDetector/dataset/models/model')
  return learn


# In[28]:


def suggestlr():
  learn.lr_find()


# In[29]:


reduceLR = ReduceLROnPlateau(monitor='accuracy')
savior = SaveModelCallback(monitor='accuracy',with_opt=True)


# In[30]:


def trainmodel():
  learn.fine_tune(20, 1e-4, cbs=[reduceLR,savior])


# In[31]:


learner = get_model()


# In[14]:


# learn.export('maskmodel.pkl')


# In[34]:


import os
import cv2
def returnfaces(imgpath):
  facecascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
  try:
    img = cv2.imread(imgpath)
  except:
    img = imgpath
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  faces = facecascade.detectMultiScale(
      img,
      scaleFactor=1.3,
      minNeighbors=3,
      minSize=(30, 30)
  )
  facelist = list()
  regions = list()
  for (x, y, w, h) in faces:
      #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
      roi_color = img[y:y + h, x:x + w, :]
      facelist.append(roi_color)
      regions.append((x,y,w,h))
  #print("Number of faces detected: "+str(len(facelist)))
  return facelist, regions


# In[35]:


def predict_facemasks(learner, imgpath):
  facelist, rects = returnfaces(imgpath)
  try:
    img = cv2.imread(imgpath)
  except:
    img = imgpath
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  positives = []
  negatives = []
  for i,f in enumerate(facelist):
    x, y, w, h = rects[i]
    if learner.predict(f)[1].item()==0:
      cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)
      positives.append(f)
    else:
      cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
      negatives.append(f)

  return img, positives, negatives


# In[36]:


get_ipython().system('pip install -Uqq ipywidgets')


# In[37]:


#hide_output
from ipywidgets import *
btn_upload = widgets.FileUpload()


# In[38]:


try:
  img = PILImage.create(btn_upload.data[-1])
except:
  pass
in_pl = widgets.Output()
in_pl.clear_output()
try:
  with in_pl: display(img.to_thumb(512,512))
except:
  pass


# In[44]:


try:
    img = PILImage.create(btn_upload.data[-1])
except:
    pass
out_pl = widgets.Output()
out_pl.clear_output()
with out_pl: display(to_image(img).to_thumb(512,512))


# In[45]:


btn_run = widgets.Button(description='Detect Masks')


# In[46]:


#hide_output
lbl_pred = widgets.Label()
lbl_pred.value = f'Upload an image and click Detect Masks'


# In[47]:


def on_click_detect_masks(change):
  img, pos, neg = predict_facemasks(learner,np.asarray(PILImage.create(btn_upload.data[-1])))
  # finalimage = PILImage.create(img)
  finalimage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  finalimage = to_image(finalimage)
  out_pl.clear_output()
  with out_pl: display(finalimage.to_thumb(512,512))
  lbl_pred.value = f'{len(pos)} faces detected wearing masks out of {len(pos)+len(neg)}'

btn_run.on_click(on_click_detect_masks)


# In[48]:


#hide_output
VBox([widgets.Label('Upload your image!'), 
      btn_upload, btn_run,
      widgets.Label('Input:'),
      in_pl, 
      widgets.Label('Output:'),
      out_pl, lbl_pred])


# In[ ]:





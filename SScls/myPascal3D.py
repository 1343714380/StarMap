import torch.utils.data as data
import torch
import numpy as np
import ref
import torch
from h5py import File
import cv2
from utils.utils import Rnd, Flip
from utils.img import Crop, DrawGaussian, Transform, Transform3D

class Pascal3D(data.Dataset):
  def __init__(self, opt, split):
    print('==> initializing pascal3d Star {} data.'.format(split))
    annot = {}
    tags = ['bbox', 'anchors', 'vis', 'dataset', 'class_id', 'imgname', 
            'viewpoint_azimuth', 'viewpoint_elevation', 'viewpoint_theta', 'anchors_3d', 
            'space_embedding', 'truncated', 'occluded', 'difficult','valid']
    f = File('{}/testPascal3D/Pascal3D-{}.h5'.format(ref.dataDir, split), 'r')
    for tag in tags:
      annot[tag] = np.asarray(f[tag]).copy()
    f.close()
    annot['index'] = np.arange(len(annot['class_id']))
    tags = tags + ['index']

    self.split = split
    self.opt = opt
    self.annot = annot
    self.nSamples = len(annot['class_id'])
    print('Loaded Pascal3D {} {} samples'.format(split, self.nSamples))
  
  def load_tags_from_h5(self,tags,phase,split):
    if(phase == 'label'):
      f = File('{}/testPascal3D/ulbPascal3D.h5'.format(ref.dataDir), 'r')
    elif(phase =='train'):
      f = File('{}/testPascal3D/Pascal3D-{}.h5'.format(ref.dataDir, split), 'r')
    else:
      f = File('{}/testPascal3D/Pascal3D-{}.h5'.format(ref.dataDir, split), 'r')
      f2 =  File('{}/testPascal3D/pseudoPascal3D-{}.h5'.format(ref.dataDir, split), 'r')
    annot = {}  
    for tag in tags:
      annot[tag] = np.asarray(f[tag]).copy()
    f.close()

    if(phase == 'ulb_train'):
      for tag in tags:
        annot[tag].append(np.asarray(f2[tag]).copy())
      f2.close()
    
    annot['index'] = np.arange(len(annot['class_id']))
    tags = tags + ['index']
    return annot

  def filter_occluded_truncated_during_val(self,annot,split,tags):
    inds = []
    if split == 'train':
      inds = np.arange(len(annot['class_id']))
    else:
      inds = []
      for i in range(len(annot['class_id'])):
        if annot['truncated'][i] < 0.5 and annot['occluded'][i] < 0.5 and annot['difficult'][i] < 0.5:
          inds.append(i)
    for tag in tags:
      annot[tag] = annot[tag][inds]
    return annot

  def LoadImage(self, index):
    img_name = ''
    for v in range(len(self.annot['imgname'][index])):
      c = self.annot['imgname'][index][v]
      if c != 0:
        img_name += chr(c)
    path = '{}/Images/{}_{}/{}'.format(ref.pascal3dDir, ref.pascalClassName[self.annot['class_id'][index]], 
                                ref.pascalDatasetName[self.annot['dataset'][index]], img_name)
    img = cv2.imread(path)
    return img, img_name
  
  
  def GetPartInfo(self, index):
    box = self.annot['bbox'][index].copy()
    c = np.array([(box[0] + box[2]) / 2., (box[1] + box[3]) / 2.])
    s = max((box[2] - box[0]), (box[3] - box[1])) * ref.padScale
    v = np.array([self.annot['viewpoint_azimuth'][index], self.annot['viewpoint_elevation'][index], 
         self.annot['viewpoint_theta'][index]]) / 180.
    #range of v:(-1,1)
    return c, s, v

  def random_crop(self, img, c, s, v):
    s = s * (2 ** Rnd(ref.scale))
    c[1] = c[1] + Rnd(ref.shiftY)
    r = 0 if np.random.random() < 0.6 else Rnd(ref.rotate)
    v[2] += r / 180.
    v[2] += 2 if v[2] < -1 else (-2 if v[2] > 1 else 0)  
    img = Crop(img, c, s, r, ref.inputRes)
    return img

  def random_flip(self,inp,v):
    if np.random.random() < 0.5:
        inp = Flip(inp)
        v[0] = - v[0]
        v[2] = - v[2]
        v[2] += 2 if v[2] <= -1 else 0
    return inp, v
  
  def view_discretization(self, v):
    #https://github.com/shubhtuls/ViewpointsAndKeypoints/blob/master/rcnnVp/rcnnBinnedJointTrainValTestCreate.m#L77
    vv = v.copy()
    if vv[0] < 0:
      v[0] = self.opt.numBins - 1 - np.floor(-vv[0] * self.opt.numBins / 2.)
    else:
      v[0] = np.floor(vv[0] * self.opt.numBins / 2.)
    v[1] = np.ceil(vv[1] * self.opt.numBins / 2. + self.opt.numBins / 2. - 1)
    v[2] = np.ceil(vv[2] * self.opt.numBins / 2. + self.opt.numBins / 2. - 1)
    v = v.astype(np.int32)

  def making_category_specific_label(self,v,class_id):
    vv = np.ones(3 * len(ref.pascalClassId), dtype = np.int32) * self.opt.numBins
    vv[class_id * 3: class_id * 3 + 3] = v.copy()
    v = vv.copy()
    return v


  def __getitem__(self, index):
    img,img_name = self.LoadImage(index)
    class_id = self.annot['class_id'][index]
    c, s, v = self.GetPartInfo(index)
    s = min(s, max(img.shape[0], img.shape[1])) * 1.0

    r = 0
    if self.split == 'train':
      inp = self.random_crop(img, c, s, v)
      inp = inp.transpose(2, 0, 1).astype(np.float32) / 256.
      inp,v = self.random_flip(inp,v)
    else :
      inp = Crop(img, c, s, r, ref.inputRes)
      inp = inp.transpose(2, 0, 1).astype(np.float32) / 256.
    

    self.view_discretization(v)

    #把其他类的view置为numBins
    if self.opt.specificView:
      v = self.making_category_specific_label(v,class_id)



    return dict(img= inp, annot=v,img_name = img_name)

    
  def __len__(self):
    return self.nSamples

def get_dataloader(opt):
  train_loader = data.DataLoader(
  Pascal3D(opt, 'train'), 
  batch_size = opt.trainBatch, 
  shuffle = True,
  num_workers = int(opt.nThreads)
  )
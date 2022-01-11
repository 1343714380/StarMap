import h5py
import torch
import numpy as np
from utils.utils import AverageMeter, Flip
from utils.eval import AccViewCls
from utils.hmParser import parseHeatmap
import cv2
import ref
from progress.bar import Bar
from utils.debugger import Debugger

def step(split, epoch, opt, dataLoader, model, criterion, optimizer = None):
  if split == 'train':
    model.train()
  else:
    model.eval()
  preds = []
  Loss, Acc = AverageMeter(), AverageMeter()
  
  nIters = len(dataLoader)
  bar = Bar('{}'.format(opt.expID), max=nIters)
  
  pseudo_label = {}
  for i, data in enumerate(dataLoader):
    input = data['img']
    input_var = torch.autograd.Variable(input.cuda(opt.GPU, True)).float().cuda(opt.GPU)
    output = model(input_var)
    numBins = opt.numBins

    if opt.phase == 'label':
      pseudo_batch = filter_label(data,opt,numBins,output)
      for key in pseudo_batch:
        pseudo_label[key]+= pseudo_batch[key]
      bar.next()
      continue

    else:
      view = data['annot']
    
    
    #(B,3*12)->(B*3*12)
    target_var = torch.autograd.Variable(view.view(-1)).long().cuda(opt.GPU)

    
    # let other label = numBins
    loss =  torch.nn.CrossEntropyLoss(ignore_index = numBins).cuda(opt.GPU)(output.view(-1, numBins), target_var)

    Acc.update(AccViewCls(output.data, view, numBins, opt.specificView))
    Loss.update(loss.item(), input.size(0))

    if split == 'train':
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    else:
      if opt.test:
        out = {}
        input_ = input.cpu().numpy()
        input_[0] = Flip(input_[0]).copy()
        inputFlip_var = torch.autograd.Variable(torch.from_numpy(input_).view(1, input_.shape[1], ref.inputRes, ref.inputRes)).float().cuda(opt.GPU)
        outputFlip = model(inputFlip_var)
        pred = outputFlip.data.cpu().numpy()
        numBins = opt.numBins
        
        if opt.specificView:
          nCat = len(ref.pascalClassId)
          pred = pred.reshape(1, nCat, 3 * numBins)
          azimuth = pred[0, :, :numBins]
          elevation = pred[0, :, numBins: numBins * 2]
          rotate = pred[0, :, numBins * 2: numBins * 3]
          azimuth = azimuth[:, ::-1]
          rotate = rotate[:, ::-1]
          output_flip = []
          for c in range(nCat):
            output_flip.append(np.array([azimuth[c], elevation[c], rotate[c]]).reshape(1, numBins * 3))
          output_flip = np.array(output_flip).reshape(1, nCat * 3 * numBins)
        else:
          azimuth = pred[0][:numBins]
          elevation = pred[0][numBins: numBins * 2]
          rotate = pred[0][numBins * 2: numBins * 3]
          azimuth = azimuth[::-1]
          rotate = rotate[::-1]
          output_flip = np.array([azimuth, elevation, rotate]).reshape(1, numBins * 3)
        out['reg'] = (output.data.cpu().numpy() + output_flip) / 2.
        preds.append(out)
 
    Bar.suffix = '{split:5} Epoch: [{0}][{1}/{2}]| Total: {total:} | ETA: {eta:} | Loss {loss.avg:.6f} | Acc {Acc.avg:.6f}'.format(epoch, i, nIters, total=bar.elapsed_td, eta=bar.eta_td, loss=Loss, Acc=Acc, split = split)
    bar.next()
  bar.finish()
  
  if(opt.phase == 'label'):
    return pseudo_label
  
  return {'Loss': Loss.avg, 'Acc': Acc.avg}, preds

def train(epoch, opt, train_loader, model, criterion, optimizer):
  return step('train', epoch, opt, train_loader, model, criterion, optimizer)
  
def val(epoch, opt, val_loader, model, criterion):
  return step('val', epoch, opt, val_loader, model, criterion)

def filter_label(data,opt,numBins,output):
  img_name = data['imgname']
  class_id = data['class_id']
  output.view(opt.trainBatch,-1,numBins)
  view = torch.rand(output.shape[0],3, numBins)
  if(opt.specificView):
    for i in range(output.shape[0]):
      id = class_id[i]
      view[i] = output[i][3*id : 3*id + 3]
  else:
    view = output
  confidence,view = torch.max(view,dim = -1)
  mark = confidence > opt.thres
  
  pseudo_img=[]
  pseudo_class = []
  azimuth = []
  elevation =[]
  theta= []
  for i in range(mark.shape[0]):
    if(mark[i][0] and mark[i][1] and mark[i][2]):
      pseudo_img.append(img_name[i])
      pseudo_class.append(class_id[i])
      azimuth.append(view[i][0])
      elevation.append(view[i][1])
      theta.append(view[i][2])

  return dict(img_name = pseudo_img, 
              cls = pseudo_class, 
              viewpoint_azimuth= azimuth,
              viewpoint_elevation = elevation,
              viewpoint_theta = theta
  )

def label_train_set(opt, train_loader, model):
  annot = step('val', 0, opt, train_loader, model, None) # set split='val', let the mode be eval mode
  load_tags_to_h5(annot,'train')

def label_val_set(opt, val_loader, model):
  annot = step('val', 0, opt, val_loader, model, None)
  load_tags_to_h5(annot,'val')

def load_tags_to_h5(annot,split):
  tags = ['class_id', 'imgname', 'viewpoint_azimuth', 'viewpoint_elevation', 'viewpoint_theta']
  nSamples = len(annot['imgname'])
  print('Save Pascal3D pseudo label data{} samples'.format(nSamples))
  with h5py.File('{}/testPascal3D/pseudoPascal3D-{}.h5'.format(ref.dataDir, split),'w') as f:
    for tag in tags:
      f[tag]=annot[tag].copy()
  f.close()
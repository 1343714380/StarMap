import torchvision.models as models
import ref
import torch
import torch.nn as nn
import os
import torchvision.models as models
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

#Re-init optimizer
def getModel(opt): 
  if opt.unpretrain:
    print("=> using unpretrained model '{}'".format(opt.arch))
  else:
    print("=> using pre-trained model '{}'".format(opt.arch))
  model = models.__dict__[opt.arch](pretrained= not opt.unpretrain)
  if opt.arch.startswith('resnet'):
    model.avgpool = nn.AvgPool2d(8, stride=1)
    if '18' in opt.arch:
      model.fc = nn.Sequential(nn.Linear(512 * 1, opt.numOutput))
      #softmax will cause to fault if use specificView
    else :
      model.fc = nn.Sequential(nn.Linear(512 * 4, opt.numOutput))

    """
    nn.init.kaiming_normal_(model.fc.weight)
    if model.fc.bias is not None:
      nn.init.constant_(model.fc.bias, 0)
    """
      
  if opt.arch.startswith('densenet'):
    if '161' in opt.arch:
      model.classifier = nn.Linear(2208, opt.numOutput)
    elif '201' in opt.arch:
      model.classifier = nn.Linear(1920, opt.numOutput)
    else:
      model.classifier = nn.Linear(1024, opt.numOutput)
  if opt.arch.startswith('vgg'):
    feature_model = list(model.classifier.children())
    feature_model.pop()
    feature_model.append(nn.Linear(4096, opt.numOutput))
    model.classifier = nn.Sequential(*feature_model)
  optimizer = torch.optim.SGD(model.parameters(), opt.LR,
                          momentum=0.9,
                          weight_decay=1e-4)
  
  if opt.loadModel:
    print("=> loading model '{}'".format(opt.cpt))
    checkpoint = torch.load(opt.cpt)
    if type(checkpoint) == type({}):
      state_dict = checkpoint['state_dict']
    else:
      state_dict = checkpoint.state_dict()
    
    model.load_state_dict(state_dict)
  return model, optimizer
  
def saveModel(path, model, optimizer = None):
  if optimizer is None:
    torch.save({'state_dict': model.state_dict()}, path)
  else:
    torch.save({'state_dict': model.state_dict(), 
                'optimizer': optimizer.state_dict()}, path)

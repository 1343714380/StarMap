import argparse
import os
import sys
import ref

class opts():
  def __init__(self):
    self.parser = argparse.ArgumentParser()
    self.parser.add_argument('-expID', default = 'cls', help = 'Experiment ID')
    self.parser.add_argument('-GPU', type = int, default = 0, help = 'GPU id')
    self.parser.add_argument('-nThreads', type = int, default = 2, help = 'nThreads')
    self.parser.add_argument('-dataset', default = 'Pascal3D', help = 'Pascal3D | ObjectNet3D')
    self.parser.add_argument('-test', action = 'store_true', help = 'test')
    self.parser.add_argument('-DEBUG', type = int, default = 0, help = 'DEBUG level')
    self.parser.add_argument('-demo', default = '', help = 'path/to/demo/image')
    self.parser.add_argument('-debugPath', default = '../debug/', help = '')
    self.parser.add_argument('-saveAllModels', action = 'store_true', help = '')
    
    self.parser.add_argument('-cpt',
    default='/home/kpl/jr/StarMap/exp/cls/ssraio0.05/sofa/ssratio_0.05_resnet50_trainingPhase_GeneralView_Pretrained_model_cpu.pth')
    self.parser.add_argument('-loadModel', action='store_true',help = 'Provide full path to a previously trained model')
    self.parser.add_argument('-arch', default = 'resnet50', help = '')
    self.parser.add_argument('-nFeats', type = int, default = 256, help = '# features in the hourglass')
    self.parser.add_argument('-nStack', type = int, default = 2, help = '# hourglasses to stack')
    self.parser.add_argument('-nModules', type = int, default = 2, help = '# residual modules at each hourglass')
    
    self.parser.add_argument('-LR', type = float, default = 0.01, help = 'Learning Rate')
    self.parser.add_argument('-dropLR', type = int, default = 20, help = 'drop LR')
    self.parser.add_argument('-nEpochs', type = int, default = 90, help = '#training epochs')
    self.parser.add_argument('-valIntervals', type = int, default = 5, help = '#valid intervel')
    self.parser.add_argument('-trainBatch', type = int, default = 32, help = 'Mini-batch size')
    self.parser.add_argument('-regWeight', type = float, default = 0.1, help = 'depth regression loss weight')
    self.parser.add_argument('-task', default = 'cls', help = 'star | staremb | starembdep | cls')
    self.parser.add_argument('-numBins', type = int, default = 21, help = '')
    self.parser.add_argument('-specificView', action = 'store_true', help = '')
    self.parser.add_argument('-ObjectNet3DTrainAll', action = 'store_true', help = '')

    self.parser.add_argument('-phase', default='train', help = 'train | ulb_train | label')
    self.parser.add_argument('-thres', default=0.9, help = '')
    self.parser.add_argument('-ssratio', default=0.05, help = '')
    self.parser.add_argument('-single_cate', action='store_true', help = '')
    self.parser.add_argument('-cate', default= 'sofa', help = '')
    self.parser.add_argument('-testAccu', action='store_true', help = '')
    self.parser.add_argument('-unpretrain', action='store_true', help = '')

  def parse(self):
    opt = self.parser.parse_args()
    if opt.DEBUG > 0:
      ref.nThreads = 1
    
    opt.numOutput = 1
    if 'emb' in opt.task:
      opt.numOutput += 3
    if 'dep' in opt.task:
      opt.numOutput += 1

    if opt.task == 'cls':
      assert not ('hg' in opt.arch), 'architecture for classification should not be hg!'
      opt.numOutput = opt.numBins * 3
      # train classifer per class output= bin * 3 * num_class
      if opt.specificView:
        opt.expID +='Spec'
        opt.numOutput *= len(ref.pascalClassId)

    print('Output dim', opt.numOutput)
    if opt.demo != '':
      opt.GPU = -1
      return opt
    if opt.test:
      opt.expID = opt.expID + 'TEST'
    opt.saveDir = os.path.join(ref.expDir, opt.expID,'ssraio'+str(opt.ssratio))
    if opt.single_cate :
      opt.saveDir = os.path.join(opt.saveDir,opt.cate)

    args = dict((name, getattr(opt, name)) for name in dir(opt)
                if not name.startswith('_'))
    refs = dict((name, getattr(ref, name)) for name in dir(ref)
                if not name.startswith('_'))
    if not os.path.exists(opt.debugPath):
      os.makedirs(opt.debugPath)
    if not os.path.exists(opt.saveDir):
      os.makedirs(opt.saveDir)
    file_name = os.path.join(opt.saveDir, 'opt.txt')
    with open(file_name, 'wt') as opt_file:
      opt_file.write('==> Cmd:\n')
      opt_file.write(str(sys.argv))
      opt_file.write('\n==> Opt:\n')
      for k, v in sorted(args.items()):
         opt_file.write('  %s: %s\n' % (str(k), str(v)))
      opt_file.write('==> Ref:\n')
      for k, v in sorted(refs.items()):
         opt_file.write('  %s: %s\n' % (str(k), str(v)))
    return opt

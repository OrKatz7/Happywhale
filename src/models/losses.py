import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# from models.triplet_loss import global_loss,TripletLoss,local_loss
from models.utils import sigmoid_loss

# class TripletLoss_sigmoid:
#     def __init__(self,margin=0.3,topk=30,w = {"tw":0.3,"sw":0.5,"lw":0.2}):
#         self.margin=margin
#         self.topk=topk
#         self.bce = nn.CrossEntropyLoss()
#         self.w = w
#     def __call__(self, global_feat, local_feat, out , species, labels,labels2):
        
#         triple_loss = global_loss(TripletLoss(margin=self.margin), global_feat, labels)[0] + \
#                           local_loss(TripletLoss(margin=self.margin), local_feat, labels)[0]
#         loss_sigmoid = sigmoid_loss(out, labels, topk=self.topk)
#         loss_class = self.bce(species,labels2)

#         return self.w['tw']*triple_loss + self.w['sw']*loss_sigmoid + self.w['lw']*loss_class
        

class LabelSmoothingLoss(nn.Module):
  def __init__(self, smoothing=0.1):
    super(LabelSmoothingLoss, self).__init__()
    self.confidence = 1.0 - smoothing
    self.smoothing = smoothing

  def forward(self, logits, labels, epoch=0, **kwargs):
    if self.training:
      logits = logits.float()
      labels = labels.float()
      logprobs = torch.nn.functional.log_softmax(logits, dim=-1)

      nll_loss = -logprobs.gather(dim=-1, index=labels.unsqueeze(1).long())
      nll_loss = nll_loss.squeeze(1)
      smooth_loss = -logprobs.mean(dim=-1)
      loss = self.confidence * nll_loss + self.smoothing * smooth_loss
      loss = loss.mean()
    else:
      loss = F.cross_entropy(logits, labels)
    return loss

class SeenBCEwithLogic(nn.Module):
    def __init__(self,*args,**kwargs):
        super(SeenBCEwithLogic,self).__init__()
        self.bce=nn.BCEWithLogitsLoss(*args,**kwargs)
    def forward(self,logits,labels,*args,**kwargs):
        return self.bce(logits,(labels>=0).to(torch.float).unsqueeze(1),*args,**kwargs)

class LabelSmoothingLossV1(nn.modules.Module):
  def __init__(self):
    super(LabelSmoothingLossV1, self).__init__()
    self.classify_loss = LabelSmoothingLoss()

  def forward(self, logits, labels, epoch=0):
    out_face, feature = logits
    loss = self.classify_loss(out_face, labels)
    return loss

# if __name__ == "__main__":
#   loss = LabelSmoothingLossV1()
#   logits = Variable(torch.randn(3, NUM_CLASSES))
#   labels = Variable(torch.LongTensor(3).random_(NUM_CLASSES))
#   output = loss([logits, None, logits], labels)
#   print(output)
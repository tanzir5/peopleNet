import torch.nn as nn
import torch

class MeanEmbRegressor(nn.Module):
  def __init__(self, emb_model):
    super(MeanEmbRegressor, self).__init__()
    self.emb_model = emb_model.to(DEVICE)
    self.regressor = nn.Linear(60, 1)
    self.loss_fn = nn.MSELoss()

  def forward(self,x,y):
    y = torch.reshape(y, (-1, 1))
    #print(y.size())
    batch_embs = None
    for input_ids in x: 
      samples_tokens_embs = self.emb_model(input_ids)[0]
      #print(samples_tokens_embs.size())
      samples_embs = torch.mean(samples_tokens_embs, axis = 1)
      #print(samples_embs.size())
      emb = torch.mean(samples_embs, axis = 0)
      emb = torch.reshape(emb, (1, -1))
      #print(emb.size())
      if batch_embs is None:
        batch_embs = emb
      else:
        batch_embs = torch.cat((batch_embs, emb))
      #print(batch_embs.size())
    out = self.regressor(batch_embs)
    #print(out.shape)
    self.loss = self.loss_fn(out,y)
    #print(self.loss)
    return {'out':out, 'loss':self.loss}
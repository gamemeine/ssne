import torch
import torch.nn.functional as F


class HingeLoss:
    def d_loss(self, real_scores, fake_scores):
        # real_scores: D(x,y), fake_scores: D(G(z),y)
        loss_real = torch.mean(F.relu(1.0 - real_scores))
        loss_fake = torch.mean(F.relu(1.0 + fake_scores))
        return loss_real + loss_fake

    def g_loss(self, fake_scores):
        # fake_scores: D(G(z),y)
        return -torch.mean(fake_scores)

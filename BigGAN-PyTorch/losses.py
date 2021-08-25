import torch
import torch.nn.functional as F

# DCGAN loss
def loss_dcgan_dis(dis_fake, dis_real):
  L1 = torch.mean(F.softplus(-dis_real))
  L2 = torch.mean(F.softplus(dis_fake))
  return L1, L2


def loss_dcgan_gen(dis_fake):
  loss = torch.mean(F.softplus(-dis_fake))
  return loss


# Hinge Loss
def loss_hinge_dis(dis_fake, dis_real):
  loss_real = torch.mean(F.relu(1. - dis_real))
  loss_fake = torch.mean(F.relu(1. + dis_fake))
  return loss_real, loss_fake
# def loss_hinge_dis(dis_fake, dis_real): # This version returns a single loss
  # loss = torch.mean(F.relu(1. - dis_real))
  # loss += torch.mean(F.relu(1. + dis_fake))
  # return loss


def loss_hinge_gen(dis_fake):
  loss = -torch.mean(dis_fake)
  return loss

# Default to hinge loss
generator_loss = loss_hinge_gen
discriminator_loss = loss_hinge_dis


def loss_multi_class_hinge(logits, label, relu=True):
  logits_choose = torch.gather(logits, -1, label.view(-1, 1))
  if relu:
    loss = F.relu(1. - logits_choose + logits)
  else:
    loss = - logits_choose + logits
  loss = torch.masked_select(loss, torch.eye(logits.size(1), device=logits.device)[label] < 0.5).mean()
  return loss


def classifier_loss_dis(logits, label, hinge=False):
  if hinge:
    loss = loss_multi_class_hinge(logits, label)
  else:
    loss = F.cross_entropy(logits, label)
  return loss


def classifier_loss_gen(logits, label, hinge=False):
  if hinge:
    loss = loss_multi_class_hinge(logits, label, False)
  else:
    loss = F.cross_entropy(logits, label)
  return loss
from argparse import Namespace
from logging import error
from tkinter import Variable
import numpy as np
from sklearn.utils import shuffle
import torch
import torch.nn as nn
from GAME.classifier import AttackerClassifier, GAMEClassifier
from GAME.models import Generator, Discriminator,ACGAN
from torch.utils.data import DataLoader
import GAME.datasets as datasets
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import random
from tensorboardX import SummaryWriter
from tqdm.auto import trange
from rich.progress import track, Progress

class baseline():
    def __init__(self,victim_model: GAMEClassifier, attacker_model:AttackerClassifier, proxyloader:DataLoader, opt:Namespace, writer: SummaryWriter):
        self.victim_model = victim_model
        self.attacker_model = attacker_model
        self.proxyloader = proxyloader
        self.batch_size = opt.batch_size
        self.writer = writer
        self.model_dir = opt.model_dir
        self.opt = opt
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cuda = True if torch.cuda.is_available() else False

    def _getqueryloader_(self, querybudget)->DataLoader:
        cuda = True if torch.cuda.is_available() else False
        queryset = []
        assert len(self.proxyloader.dataset) >= querybudget , "querybudget(%d) larger than the dataset size(%d)"%(querybudget,len(self.proxyloader.dataset))
        for imgs, _ in self.proxyloader:
            if cuda:
                imgs = imgs.cuda()
            else:
                imgs = Variable(imgs)
            
            Vout = self.victim_model.model(imgs)
            Vout = F.softmax(Vout, dim=1)
            batch = [(a, b) for a, b in zip(imgs.cpu().detach().numpy(), Vout.cpu().detach().numpy())]
            queryset += batch
        queryset = queryset[:querybudget]
        queryloader = DataLoader(queryset, batch_size=self.batch_size, num_workers=4, shuffle=True)

        return queryloader
  

    def extract(self, querybudget:int, train_epoch=10, train_log=False):
        self.queryloader = self._getqueryloader_(querybudget)
        assert len(self.queryloader.dataset) == querybudget, "length of queryloader(%d) not equals to querybudget(%d)"%(len(self.queryloader.dataset),querybudget)
        self.attacker_model.soft_train(self.queryloader, train_epoch, lr=self.opt.lr,progress=True,train_log=train_log)



class game(baseline):

    def __init__(self, victim_model: GAMEClassifier, attacker_model: AttackerClassifier, proxyloader: DataLoader, opt: Namespace, writer: SummaryWriter,id_G=1):
        super().__init__(victim_model, attacker_model, proxyloader, opt, writer)
        self.acgan = ACGAN(self.opt,id_G=id_G)
        self.acgan.train(self.proxyloader,n_epoch=self.opt.n_epoch_gan,load=True)
        
    
    def _ada_extract_(self, acgan:ACGAN, querybudget:int, train_epoch:int, batch_size=128,loss_items=['adv','dis'],sampler_weights="random",save_samples=False):
        
        self.attacker_model.model.train()
        self.victim_model.model.eval()
        acgan.generator.train()

        self.budget_left = querybudget
        self.sampler_weights = torch.ones(acgan.n_output)
        self.queryset = []
        self.finalset = []

        print("GAME attack start!")
        with Progress() as progress:
            sampler_task = progress.add_task("GAME attack:", total=querybudget)
            iterations = 0
            while self.budget_left > 0:
                errG = 0
                if self.budget_left < batch_size:
                    batch_size = self.budget_left
                    self.budget_left = 0
                else:
                    self.budget_left -= batch_size
                
                iterations += 1
                progress.update(sampler_task, advance=batch_size)

                # generator a batch of samples with ada-policy
                noise = torch.randn(batch_size,10*self.acgan.n_output,device = self.device)
                # print(self.sampler_weights)
                sample_labels = torch.multinomial(self.sampler_weights,batch_size,replacement=True).long().to(self.device)
                # print("Generate %d new samples! (%d query budget left)"%(batch_size,self.budget_left))
                syn_samples = acgan.generator(noise,sample_labels)

                # use synthetic samples to query victim model and add to queryloader
                pred_victim = self.victim_model.model(syn_samples)
                pred_victim_softmax = F.softmax(pred_victim, dim=1)
                
                batch = [(a, b) for a, b in zip(syn_samples.cpu().detach().numpy(), pred_victim_softmax.cpu().detach().numpy())]
                
                # change the following line to conduct fair comparison
                self.queryset += batch
                # self.queryset = batch
                self.finalset += batch
                self.queryloader = DataLoader(self.queryset, batch_size=self.batch_size, num_workers=4, shuffle=True)

                # use queryloader train attacker model
                self.attacker_model.soft_train(self.queryloader,20, lr=self.opt.lr,progress=False)
                self.attacker_model.evaluate(self.writer,global_step=querybudget-self.budget_left)
                

                # update generator with attacker model
                for iteration in range(2):
                    noise = torch.randn(batch_size,10*self.acgan.n_output,device = self.device)
                    sample_labels = torch.multinomial(self.sampler_weights,batch_size,replacement=True).long().to(self.device)
                    syn_samples = acgan.generator(noise,sample_labels)
                    pred_attacker = self.attacker_model.model(syn_samples)
                    pred_attacker_softmax = F.softmax(pred_attacker,dim=1)
                    pred_attacker_logsoftmax = F.log_softmax(pred_attacker,dim=1)
                    pred_victim = self.victim_model.model(syn_samples)
                    pred_victim_softmax = F.softmax(pred_victim, dim=1)
                    errG=0
                    if len(loss_items) > 0:
                        acgan.optimizer_G.zero_grad()
                        
                        label_attacker = pred_attacker_softmax.max(1)[1]
                        
                        if 'res' in loss_items:
                            errG_adv = 0.002*pred_attacker[pred_attacker>0].sum()
                            errG += errG_adv
                            self.writer.add_scalar('res_loss', errG_adv, global_step=querybudget-self.budget_left)

                        if 'bou' in loss_items:
                            sort_pred = 0.01*torch.sort(pred_attacker, descending=True)[0]
                            errG_adv = (sort_pred[:,0]-sort_pred[:,1]).sum()
                            errG += errG_adv
                            self.writer.add_scalar('bou_loss', errG_adv, global_step=querybudget-self.budget_left)

                        if 'dif' in loss_items:
                            errG_adv = -100 * F.kl_div(pred_attacker_logsoftmax,pred_victim_softmax.detach())
                            errG += errG_adv
                            self.writer.add_scalar('dif_loss', errG_adv, global_step=querybudget-self.budget_left)
                        
                        if 'adv' in loss_items:
                            errG_adv = -10 * F.cross_entropy(pred_attacker, label_attacker.detach())
                            errG += errG_adv
                            self.writer.add_scalar('adv_loss', errG_adv, global_step=querybudget-self.budget_left)
                        
                        # if 'dis' in loss_items:
                        #     errG_dis = -1 * F.cross_entropy(pred_attacker, label_victim.detach())
                        #     errG += errG_dis
                        
                        # if 'conf' in loss_items:
                        #     errG_conf = F.cross_entropy(pred_attacker, label_attacker.detach())
                        #     errG += errG_conf

                        # print("[Generator] [errG_adv: {:.4f}] [errG_dis: {:.4f}]".format(errG_adv.item(), errG_dis.item()))

                        errG.backward()

                        acgan.optimizer_G.step()

                # update the self.sampler_weights with reward
                if sampler_weights=="unconfident":
                    confident_log = torch.ones(self.acgan.n_output,2)
                    confident_log[:,0] = 1/self.acgan.n_output
                    sample_confident = pred_victim_softmax.max(1)[0]
                    for index, confident in enumerate(sample_confident):
                        confident_log[sample_labels[index],0] += confident.cpu().detach()
                        confident_log[sample_labels[index],1] += 1
                    label_unconfident = torch.ones(self.acgan.n_output)
                    for index in range(len(confident_log)):
                        label_unconfident[index] = 1 - confident_log[index,0]/confident_log[index,1]
                    self.sampler_weights = label_unconfident
                elif sampler_weights=="confident":
                    confident_log = torch.ones(self.acgan.n_output,2)
                    confident_log[:,0] = 1/self.acgan.n_output
                    sample_confident = pred_victim_softmax.max(1)[0]
                    for index, confident in enumerate(sample_confident):
                        confident_log[sample_labels[index],0] += confident.cpu().detach()
                        confident_log[sample_labels[index],1] += 1
                    label_confident = torch.ones(self.acgan.n_output)
                    for index in range(len(confident_log)):
                        label_confident[index] = confident_log[index,0]/confident_log[index,1]
                    self.sampler_weights = label_confident
                elif sampler_weights=="random":
                    self.sampler_weights = torch.ones(acgan.n_output)
                elif sampler_weights=="deviation":
                    deviation_log = torch.ones(self.acgan.n_output,2)
                    deviation_loss = F.kl_div(pred_attacker_logsoftmax.detach(),pred_victim_softmax.detach(),reduction='none')
                    sample_deviation = torch.Tensor([i.mean() for i in deviation_loss])                        
                    for index, deviation in enumerate(sample_deviation):
                        deviation_log[sample_labels[index],0] += deviation.cpu().detach()
                        deviation_log[sample_labels[index],1] += 1
                    label_deviation = torch.ones(self.acgan.n_output)
                    for index in range(len(deviation_log)):
                        label_deviation[index] = deviation_log[index,0]/deviation_log[index,1]                        
                    self.sampler_weights = label_deviation
                else:
                    raise ValueError
                self.acgan.save_samples('%s-epoch%d'%(''.join(loss_items),iterations))
        
        self.finalloader = DataLoader(self.finalset, batch_size=self.batch_size, num_workers=4, shuffle=True)
        self.attacker_model.soft_train(self.finalloader,train_epoch-10, lr=self.opt.lr,progress=self.opt.progress)


    def extract(self, querybudget:int, loss_items:list=[], train_epoch=10,sampler_weights="random"):
        self.acgan.save_samples('initial')
        self._ada_extract_(self.acgan, querybudget, train_epoch, batch_size=self.opt.batch_size, loss_items=loss_items,sampler_weights=sampler_weights,save_samples=True)

                



                





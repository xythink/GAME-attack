from argparse import Namespace
import os
from pydoc import describe
import torch
import GAME.models as models
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from rich.progress import track

class GAMEClassifier():
    def __init__(self, architecture:str, n_channels: int, n_outputs: int, opt:Namespace):
        self.n_outputs = n_outputs
        self.n_channels = n_channels
        self.model = models.__dict__[architecture](n_channels = n_channels, n_outputs = n_outputs)
        self.cuda = True if torch.cuda.is_available() else False
        self.architecture = architecture
        self.opt = opt
    
    def reload_model(self):
        self.model = models.__dict__[self.architecture](n_channels = self.n_channels, n_outputs = self.n_outputs)
    
    def predict_list(self, x_list):
        input_set = [(i.numpy(),0) for i in x_list]
        inputloader = DataLoader(input_set, batch_size=len(x_list), num_workers=0, shuffle=True)
        for imgs, _ in inputloader:
            if len(imgs.shape) < 4:
                imgs = torch.unsqueeze(imgs, dim=0).float()

            if self.cuda:
                imgs = imgs.cuda()
            else:
                imgs = Variable(imgs)

            pred = self.model(imgs)
            pred = F.softmax(pred, dim=1)
            break
        assert pred.shape[0] == len(x_list)
        return imgs, pred

    def train(self, trainloader, testloader, n_epochs:int, writer:SummaryWriter, model_name:str,dataset_name: str, save_path:str, lr=0.002, b1=0.5, b2=0.999,load=True, progress=False):

        file_model = "victim-%s-%s-%s.pth"%(self.architecture,dataset_name,n_epochs)
        
        if load and os.path.exists(os.path.join(save_path,file_model)):
            if self.cuda:
                model_load = torch.load(os.path.join(save_path,file_model))
                self.model.load_state_dict(model_load)
                self.model.cuda()
            else:
                model_load = torch.load(os.path.join(save_path,file_model),map_location='cpu')
                self.model.load_state_dict(model_load)
            print("Victim model load success!")
            test_correct, test_acc, test_loss = self.test(testloader)
            return test_acc

        if self.cuda:
            self.model.cuda()


        criteria = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(self.model.parameters(), lr = lr, betas=(b1, b2))
        bestacc = 0
        for epoch in track(range(n_epochs),description="Train Victim:"):
            self.model.train()
            for i, (imgs, labels) in enumerate(trainloader):
                target = labels.clone()

                if self.cuda:
                    imgs, labels = imgs.cuda(), labels.long().cuda()
                else:
                    imgs, labels = Variable(imgs), Variable(labels)
                
                optimizer.zero_grad()
                output = self.model(imgs)
                loss = criteria(output, labels)

                pred = output.data.max(1)[1]
                correct = pred.cpu().eq(target).sum().item()
                acc = correct * 1.0 / len(imgs)

                loss.backward()
                optimizer.step()
                if progress:
                    print("[Epoch %d/%d] [Batch %d/%d] [loss: %f, acc:%d%%]"%(epoch, n_epochs, i, len(trainloader), loss.item(), 100*acc))
                writer.add_scalar(model_name+'_batch_train_loss', loss.item(), global_step=epoch*len(trainloader)+i)
                writer.add_scalar(model_name+'_batch_train_acc', acc, global_step=epoch*len(trainloader)+i)
            test_correct, test_acc, test_loss = self.test(testloader)
            print("Test loss: %f, Test acc:%d/%d(%.2f%%)"%(test_loss, test_correct, len(testloader.dataset), test_acc))
            writer.add_scalar(model_name+'_test_loss', test_loss, global_step=epoch)
            writer.add_scalar(model_name+'_test_acc', test_acc, global_step=epoch)
            
            if test_acc > bestacc:
                best_model = self.model.state_dict()
                bestacc = test_acc
                new_file = os.path.join(save_path,file_model)
                torch.save(best_model,new_file)
        
        bestmodel_load = torch.load(new_file)
        self.model.load_state_dict(bestmodel_load)
        test_correct, test_acc, test_loss = self.test(testloader)
        assert test_acc == bestacc , "%f != %f"%(test_acc, bestacc)
        return bestacc
            
        
    
    def test(self, testloader):
        criteria = nn.CrossEntropyLoss()
        with torch.no_grad():
            self.model.eval()
            correct = 0
            testloss = 0
            for i, (imgs, labels) in enumerate(testloader):
                imgs = Variable(imgs)
                labels = Variable(labels)
                target = labels.clone()
                if self.cuda:
                    imgs, labels = imgs.cuda(), labels.long().cuda()
                else:
                    imgs, labels = Variable(imgs), Variable(labels)
                
                output = self.model(imgs)
                testloss = testloss + criteria(output, labels).item()

                pred = output.data.max(1)[1]
                correct += pred.cpu().eq(target).sum().cpu().item()

            acc = 100. * correct / len(testloader.dataset)
            testloss = testloss / len(testloader)
            return correct, acc, testloss

class AttackerClassifier(GAMEClassifier):
    def __init__(self, architecture: str, n_channels: int, n_outputs: int, testloader:DataLoader,victim_model: GAMEClassifier,opt:Namespace):
        super().__init__(architecture, n_channels, n_outputs,opt)
        self.victim = victim_model
        self.testloader = testloader
        self.n_epoch_trained = 0


    def soft_train(self, queryloader:DataLoader, n_epochs:int, lr=0.002, b1=0.5, b2=0.999,progress=True, train_log=False):

        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.model.cuda()

        criteria = torch.nn.KLDivLoss(reduction="batchmean")

        if self.opt.optimizer == "SGD":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        elif self.opt.optimizer == "ADAM":
            optimizer = torch.optim.Adam(self.model.parameters(), lr = lr, betas=(b1, b2))
        else:
            raise ValueError
        lr_flag = 0
        for epoch in track(range(n_epochs),description="Train Attacker:",disable=not progress):
            if 2*epoch >= n_epochs and lr_flag == 0:
                lr_flag = 1
                # print("change learning_rate:%f -> %f"%(lr,0.1*lr))
                lr = lr * 0.1
                if self.opt.optimizer == "SGD":
                    optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
                elif self.opt.optimizer == "ADAM":
                    optimizer = torch.optim.Adam(self.model.parameters(), lr = lr, betas=(b1, b2))
            self.model.train()
            correct = 0
            train_loss = 0
            for i, (imgs, labels) in enumerate(queryloader):
                if len(imgs) == 1 :
                    break
                target = labels.clone()

                if cuda:
                    imgs, labels = imgs.cuda(), labels.cuda()
                else:
                    imgs, labels = Variable(imgs), Variable(labels)
                
                optimizer.zero_grad()
                output = self.model(imgs)
                output_log = F.log_softmax(output, dim=1)
                loss = criteria(output_log, labels)
                train_loss += loss
                pred = output.data.max(1)[1]
                target = target.data.max(1)[1]
                correct += pred.cpu().eq(target).sum().item()
                loss.backward()
                optimizer.step()
            train_loss /= len(queryloader)
            train_acc = correct * 1.0 / len(queryloader.dataset)

            if train_log:
                print("Epoch loss: %f, Epoch acc:%d/%d(%.2f%%)"%(train_loss, correct, len(queryloader.dataset), 100*train_acc))

    
    
    
    def evaluate(self,writer=None,global_step=None):
        self.model.eval()
        self.victim.model.eval()

        same = 0
        correct = 0
        for i, (imgs, labels) in enumerate(self.testloader):
            cuda = True if torch.cuda.is_available() else False
            if cuda:
                imgs, labels = imgs.cuda(), labels.cuda()
            else:
                imgs, labels = Variable(imgs), Variable(labels)

            attack_pred = self.model(imgs)
            attack_pred_softmax = F.softmax(attack_pred,dim=1)
            attack_labels = attack_pred_softmax.data.max(1)[1]

            victim_pred = self.victim.model(imgs)
            victim_pred_softmax = F.softmax(victim_pred,dim=1)
            victim_labels = victim_pred_softmax.data.max(1)[1]

            same += attack_labels.cpu().eq(victim_labels.cpu()).sum()
            correct += attack_labels.cpu().eq(labels.cpu()).sum()

        fidelity = 100 * same.item() / len(self.testloader.dataset)
        accuracy = 100 * correct.item() / len(self.testloader.dataset)
        if writer != None:
            writer.add_scalar('Fidelity', fidelity, global_step=global_step)
            writer.add_scalar('Accuracy', accuracy, global_step=global_step)

        return fidelity, accuracy

    
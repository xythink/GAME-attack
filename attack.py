import argparse
from GAME.classifier import GAMEClassifier, AttackerClassifier
import GAME.datasets as datasets
import GAME.methods as methods
from tensorboardX import SummaryWriter
from rich.console import Console
from rich.table import Table
from GAME.utils import clear_dir
import os

def main():

    # Init the argparser
    parser = argparse.ArgumentParser()
    parser.add_argument("--testname", type=str, default='test00', help="Name of this experiment")
    parser.add_argument("--model_dir", type=str, default='model_save', help="path to save log")
    parser.add_argument("--img_dir", type=str, default='img_save', help="path to save imgs")
    parser.add_argument("--originset", type=str, default='CIFAR10', help="dataset used to train victim model")
    parser.add_argument("--proxyset", type=str, default='CIFAR100', help="dataset used to train attacker model")
    parser.add_argument("--victim_arch", type=str, default='ResNet34', help="architecture of victim model")
    parser.add_argument("--attacker_arch", type=str, default='ResNet34', help="architecture of attacker model")
    parser.add_argument("--n_epoch_V", type=int, default=20, help="Number of epochs to train victim model")
    parser.add_argument("--batch_size", type=int, default=1024, help="Number of batch_size to train model")
    parser.add_argument("--querybudget", type=int, default=2000, help="Number of query samples used by attacker")
    parser.add_argument("--tb_dir", type=str, default='/home/data/tb_logs', help="Path to save tensorboard logs")
    parser.add_argument("--attack_train_epoch", type=int, default=40, help="Number of epochs used to train attacker model")
    parser.add_argument("--optimizer", type=str, default='SGD', help="Optimizer used to train attacker model")
    opt = parser.parse_args()
    print(opt)

    img_save_path = os.path.join(opt.img_dir,opt.testname)
    clear_dir(img_save_path)
    os.makedirs(img_save_path, exist_ok=True)

    console = Console()
    table = Table(show_header=True)
    table.add_column("Method")
    table.add_column("Fidelity(%)", justify="right")
    table.add_column("Accuracy(%)", justify="right")



    # Init original dataloader used by victim model
    originTrainloader = datasets.getloader(opt.originset,train=True,batch_size=opt.batch_size)
    originTestloader = datasets.getloader(opt.originset,train=False,batch_size=opt.batch_size)
    n_channels = datasets.datasets_dict[opt.originset]['n_channels']
    n_outputs = datasets.datasets_dict[opt.originset]['n_outputs']

    # Get victim model
    print("------------------------------------------------------------")
    
    writer_dir = os.path.join(opt.tb_dir, opt.testname+'-'+'victim')
    clear_dir(writer_dir)
    writer = SummaryWriter(log_dir=writer_dir)

    print("Victim model train start!")
    victim_model = GAMEClassifier(opt.victim_arch, n_channels, n_outputs)
    victim_acc = victim_model.train(originTrainloader,originTestloader,opt.n_epoch_V, writer,'victim_model',opt.originset,opt.model_dir,load=True)
    console.print("Victim model train finished! Model accuracy = [red]%.2f%%[/red]"%victim_acc)

    

    # Init attacker model
    attacker_list = {
        'baseline': methods.baseline,
        # 'game': methods.game,
        # 'knockoff': methods.knockoff
    }

    # Config proxy dataloader used by attacker
    proxyloader = datasets.getloader(opt.proxyset,train=True,batch_size=opt.batch_size)

    for method in attacker_list.keys():
        print("------------------------------------------------------------")
        console.print("[yellow]%s[/yellow] attack start!"%method)
        
        writer_dir = os.path.join(opt.tb_dir, opt.testname+'-'+method)
        clear_dir(writer_dir)
        writer = SummaryWriter(log_dir=writer_dir)

        attacker_model = AttackerClassifier(opt.attacker_arch, n_channels, n_outputs,originTestloader,victim_model)
        attacker = attacker_list[method](victim_model, attacker_model, proxyloader,opt,writer)
        attacker.extract(opt.querybudget,train_epoch=opt.attack_train_epoch)
        fidelity, accuracy = attacker_model.evaluate(originTestloader, victim_model)
        
        console.print("[yellow]%s[/yellow] attack finished, fidelity = [red]%.2f%%[/red], accuracy = [red]%.2f%%[/red]"%(method, fidelity, accuracy))
        table.add_row(
            "[yellow]%s[/yellow]"%method,
            "[red]%.2f[/red]"%fidelity,
            "[red]%.2f[/red]"%accuracy
            )
    
    console.print(table)

            

    
if __name__ == '__main__':
    main() 
    





from argparse import ArgumentParser
from clinical_ts.simclr_dataset_wrapper import SimCLRDataSetWrapper
from clinical_ts.create_logger import create_logger
from ecg_datamodule import ECGDataModule
import logging
import math
from models.resnet_simclr import ResNetSimCLR
from models.onelayer_linear import OneLayerLinear
import numpy as np
from online_evaluator import SSLOnlineEvaluator
import os
import pdb
import pickle
from pl_bolts.models.self_supervised.evaluator import Flatten
from pl_bolts.optimizers.lars_scheduling import LARSWrapper
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import AMPType
import re
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.optim import Adam, SGD
import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
import time
from typing import Callable, Optional
import yaml

method="simclr"
logger = create_logger(__name__)
def _accuracy(zis, zjs, batch_size):
    with torch.no_grad():
        representations = torch.cat([zjs, zis], dim=0)
        similarity_matrix = torch.mm(
            representations, representations.t().contiguous())
        corrected_similarity_matrix = similarity_matrix - \
            torch.eye(2*batch_size).type_as(similarity_matrix)
        pred_similarities, pred_indices = torch.max(
            corrected_similarity_matrix[:batch_size], dim=1)
        correct_indices = torch.arange(batch_size)+batch_size
        correct_preds = (
            pred_indices == correct_indices.type_as(pred_indices)).sum()
    return correct_preds.float()/batch_size

def mean(res, key1, key2=None):
    if key2 is not None:
        return torch.stack([x[key1][key2] for x in res]).mean()
    return stack(res, key1).mean()

def stack(res, key1):
    return torch.stack([x[key1] for x in res if type(x) == dict and key1 in x.keys()])

def cat(res, key1):
    return torch.cat([x[key1] for x in res if type(x) == dict and key1 in x.keys()], 0)

class Projection(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=2048, output_dim=128):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        if hidden_dim:
            self.model = nn.Sequential(
                # nn.AdaptiveAvgPool2d((1, 1)),
                Flatten(),
                nn.Linear(self.input_dim, self.hidden_dim, bias=True),
                # nn.BatchNorm1d(self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.output_dim, bias=True))
        else:
            self.model = nn.Sequential(
                Flatten(),
                nn.Linear(self.input_dim, self.output_dim, bias=True))

    def forward(self, x):
        x = self.model(x)
        return F.normalize(x, dim=1)


class SyncFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tensor):
        ctx.batch_size = tensor.shape[0]

        gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]

        torch.distributed.all_gather(gathered_tensor, tensor)
        gathered_tensor = torch.cat(gathered_tensor, 0)

        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        torch.distributed.all_reduce(grad_input, op=torch.distributed.ReduceOp.SUM, async_op=False)

        return grad_input[torch.distributed.get_rank() * ctx.batch_size:(torch.distributed.get_rank() + 1) *
                          ctx.batch_size]


class CustomSimCLR(pl.LightningModule):

    def __init__(self,
                 batch_size,
                 num_samples,
                 warmup_epochs=10,
                 lr=1e-4,
                 opt_weight_decay=1e-6,
                 loss_temperature=0.5,
                 config=None,
                 transformations=None,
                 cls_normalizer=None, # None | "std" | "norm"
                 optimizer_name='Adam',
                 **kwargs):
        """
        Args:
            batch_size: the batch size
            num_samples: num samples in the dataset
            warmup_epochs: epochs to warmup the lr for
            lr: the optimizer learning rate
            opt_weight_decay: the optimizer weight decay
            loss_temperature: the loss temperature
        """

        super(CustomSimCLR, self).__init__()
        self.config = config
        self.transformations = transformations
        self.epoch = 0
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.cls_normalizer = cls_normalizer
        self.optimizer_name = optimizer_name
        self.save_hyperparameters()

    def configure_optimizers(self):
        # TRICK 1 (Use lars + filter weights)
        # exclude certain parameters
        parameters = self.exclude_from_wt_decay(
            self.named_parameters(),
            weight_decay=self.hparams.opt_weight_decay
        )
        if self.optimizer_name == 'Adam':
            global_batch_size = self.trainer.world_size * self.hparams.batch_size
            self.train_iters_per_epoch = self.hparams.num_samples // global_batch_size
            
            # optimizer = LARSWrapper(Adam(parameters, lr=self.hparams.lr))
            optimizer = Adam(parameters, lr=self.hparams.lr)
            
            # Trick 2 (after each step)
            self.hparams.warmup_epochs = self.hparams.warmup_epochs * self.train_iters_per_epoch
            max_epochs = self.trainer.max_epochs * self.train_iters_per_epoch

            # TODO(loganesian): Enable different scheduler for linear model.
            linear_warmup_cosine_decay = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=self.hparams.warmup_epochs,
                max_epochs=max_epochs,
                warmup_start_lr=0,
                eta_min=0
            )

            scheduler = {
                'scheduler': linear_warmup_cosine_decay,
                'interval': 'step',
                'frequency': 1
            }
            return [optimizer], [scheduler]

        # else: optimizer_name == SGD
        # weight_decay = self.hparams.opt_weight_decay??
        optimizer = SGD(parameters, lr=self.hparams.lr, weight_decay=0.0)
        return [optimizer], []

    def exclude_from_wt_decay(self, named_params, weight_decay, skip_list=['bias', 'bn']):
        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [
            {'params': params, 'weight_decay': weight_decay},
            {'params': excluded_params, 'weight_decay': 0.}
        ]
    
    def shared_forward(self, batch, batch_idx):
        (x1, y1), (x2, y2) = batch
        # ENCODE
        # encode -> representations
        # (b, 3, 32, 32) -> (b, 2048, 2, 2)
        x1 = self.to_device(x1)
        x2 = self.to_device(x2)

        h1 = self.encoder(x1)[0]
        h2 = self.encoder(x2)[0]

        # the bolts resnets return a list of feature maps
        if isinstance(h1, list):
            h1 = h1[-1]
            h2 = h2[-1]

        # PROJECT
        # img -> E -> h -> || -> z
        # (b, 2048, 2, 2) -> (b, 128)
        z1 = self.projection(h1.squeeze())
        z2 = self.projection(h2.squeeze())

        return z1, z2

    def nt_xent_loss(self, out_1, out_2, temperature, eps=1e-6):
        """
            assume out_1 and out_2 are normalized
            out_1: [batch_size, dim]
            out_2: [batch_size, dim]
        """
        # gather representations in case of distributed training
        # out_1_dist: [batch_size * world_size, dim]
        # out_2_dist: [batch_size * world_size, dim]
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            out_1_dist = SyncFunction.apply(out_1)
            out_2_dist = SyncFunction.apply(out_2)
            print("out dist shape: ", out_1_dist.shape)
        else:
            out_1_dist = out_1
            out_2_dist = out_2
        
        # out: [2 * batch_size, dim]
        # out_dist: [2 * batch_size * world_size, dim]
        out = torch.cat([out_1, out_2], dim=0)
        out_dist = torch.cat([out_1_dist, out_2_dist], dim=0)

        # cov and sim: [2 * batch_size, 2 * batch_size * world_size]
        # neg: [2 * batch_size]
        cov = torch.mm(out, out_dist.t().contiguous())
        sim = torch.exp(cov / temperature)
        neg = sim.sum(dim=-1)

        # from each row, subtract e^1 to remove similarity measure for x1.x1
        row_sub = torch.Tensor(neg.shape).fill_(math.e).to(neg.device)
        neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

        # Positive similarity, pos becomes [2 * batch_size]
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        pos = torch.cat([pos, pos], dim=0)

        loss = -torch.log(pos / (neg + eps)).mean()

        return loss

    def on_train_start(self):
        # log configuration
        config_str = re.sub(r"[,\}\{]", "<br/>", str(self.config))
        config_str = re.sub(r"[\[\]\']", "", config_str)
        transformation_str = re.sub(r"[\}]", "<br/>", str(["<br>" + str(
            t) + ":<br/>" + str(t.get_params()) for t in self.transformations]))
        transformation_str = re.sub(r"[,\"\{\'\[\]]", "", transformation_str)
        self.logger.experiment.add_text(
            "configuration", str(config_str), global_step=0)
        self.logger.experiment.add_text("transformations", str(
            transformation_str), global_step=0)
        self.epoch = 0

    def training_step(self, batch, batch_idx):
        z1, z2 = self.shared_forward(batch, batch_idx)
        loss = self.nt_xent_loss(z1, z2, self.hparams.loss_temperature)
        acc = _accuracy(z1, z2, z1.shape[0])
        result = {
            "loss": loss,
            "acc" : acc,
        }
        return result

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["acc"] for x in outputs]).mean()

        log = {"train/train_loss": avg_loss, "train/train_acc": avg_acc}
        self.logger.experiment.add_scalar("train/train_loss", avg_loss, self.epoch)
        self.logger.experiment.add_scalar("train/train_acc", avg_acc, self.epoch)

    def on_train_epoch_end(self):
        self.epoch += 1

    def validation_step(self, batch, batch_idx, dataloader_idx):
        results = {}
        if dataloader_idx != 0: return results
        z1, z2 = self.shared_forward(batch, batch_idx)
        loss = self.nt_xent_loss(z1, z2, self.hparams.loss_temperature)
        acc = _accuracy(z1, z2, z1.shape[0])
        results["val_loss"] = loss
        results["val_acc"] = torch.tensor(acc)
        return results

    def validation_epoch_end(self, outputs):
        # outputs[0] because we are using multiple datasets!
        val_loss = mean(outputs[0], "val_loss")
        val_acc = mean(outputs[0], "val_acc")

        log = {"val/val_loss": val_loss, "val/val_acc": val_acc}
        self.logger.experiment.add_scalar("val/val_loss", val_loss, self.epoch)
        self.log("val/val_loss", val_loss)
        self.logger.experiment.add_scalar("val/val_acc", val_acc, self.epoch)
        results = {"val_loss": val_loss, "val_acc": val_acc}
        results["log"] = results["progress_bar"] = log
        return results

    def type(self):
        if hasattr(self.encoder, 'features'):
            return self.encoder.features[0][0].weight.type()
        else:
            return self.encoder.l1.weight.type()

    def get_representations(self, x):
        return self.encoder(x)[0]
    
    def get_model(self):
        return self.encoder

    def get_device(self):
        if hasattr(self.encoder, 'features'):
            return self.encoder.features[0][0].weight.device
        else:
            return self.encoder.l1.weight.device

    def to_device(self, x):
        return x.type(self.type()).to(self.get_device())

def parse_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument('--config_name', default='bolts_config.yaml')
    parser.add_argument('-t', '--trafos', nargs='+', help='add transformation to data augmentation pipeline',
                        default=["GaussianNoise", "ChannelResize", "RandomResizedCrop", "Normalize"])
    # GaussianNoise
    parser.add_argument(
            '--gaussian_scale', help='std param for gaussian noise transformation', default=0.01, type=float) # 0.005 original default
    # RandomResizedCrop
    parser.add_argument('--rr_crop_ratio_range',
                            help='ratio range for random resized crop transformation', default=[0.5, 1.0], type=float)
    parser.add_argument(
            '--output_size', help='output size for random resized crop transformation', default=250, type=int)
    # DynamicTimeWarp
    parser.add_argument(
            '--warps', help='number of warps for dynamic time warp transformation', default=3, type=int)
    parser.add_argument(
            '--radius', help='radius of warps of dynamic time warp transformation', default=10, type=int)
    # TimeWarp
    parser.add_argument(
            '--epsilon', help='epsilon param for time warp', default=10, type=float)
    # ChannelResize
    parser.add_argument('--magnitude_range', nargs='+',
                            help='range for scale param for ChannelResize transformation', default=[0.5, 2], type=float)
    # Downsample
    parser.add_argument(
            '--downsample_ratio', help='downsample ratio for Downsample transformation', default=0.2, type=float)
    # TimeOut
    parser.add_argument('--to_crop_ratio_range', nargs='+',
                            help='ratio range for timeout transformation', default=[0.2, 0.4], type=float)
    # resume training
    parser.add_argument('--resume', action='store_true')
    parser.add_argument(
            '--gpus', help='number of gpus to use; use cpu if gpu=0', type=int, default=1)
    parser.add_argument(
            '--num_nodes', default=1, help='number of cluster nodes', type=int)
    parser.add_argument(
            '--distributed_backend', help='sets backend type')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--warm_up', default=1, type=int, help="number of warm up epochs")
    parser.add_argument('--precision', type=int)
    parser.add_argument('--datasets', dest="target_folders",
                            nargs='+', help='used datasets for pretraining')
    parser.add_argument('--log_dir', default="./experiment_logs")
    parser.add_argument(
            '--percentage', help='determines how much of the dataset shall be used during the pretraining', type=float, default=1.0)
    parser.add_argument('--lr', type=float, help="learning rate")
    parser.add_argument('--out_dim', type=int, help="output dimension of model")
    parser.add_argument('--filter_cinc', default=False, action="store_true", help="only valid if cinc is selected: filter out the ptb data")
    parser.add_argument('--base_model')
    parser.add_argument('--ftr_multiplier', default=1, type=float,
                        help='Amount to multiply number of training samples by to define number of features.')
    parser.add_argument('--activation', type=str, default="F.relu", help='Optional activation for linear models.')
    parser.add_argument('--optimizer_name', type=str, default="Adam",
                        choices=["Adam", "SGD"], help='Optimizer to use.')
    parser.add_argument('--force_linear_projection', type=bool, default=False,
                        help='OneLayerLinear models only: use linear w/ no hidden layers.')
    parser.add_argument('--widen',type=int, help="use wide xresnet1d50")
    parser.add_argument('--run_callbacks', default=False, action="store_true", help="run callbacks which asses linear evaluaton and finetuning metrics during pretraining")
    parser.add_argument('--early_stopping', default=False, action="store_true", help="enable early stopping based on validation cross entropy loss")
    parser.add_argument('--best_model_ckpt', default=True, action="store_true", help="enable best model chkpt saving")
    parser.add_argument('--checkpoint_path', default="")
    return parser

def init_logger(config):
    level = logging.INFO

    if config['debug']:
        level = logging.DEBUG

    # remove all handlers to change basic configuration
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    if not os.path.isdir(config['log_dir']):
        os.mkdir(config['log_dir'])
    logging.basicConfig(filename=os.path.join(config['log_dir'], 'info.log'), level=level,
                        format='%(asctime)s %(name)s:%(lineno)s %(levelname)s:  %(message)s  ')
    return logging.getLogger(__name__)

def pretrain_routine(args):
    t_params = {"gaussian_scale": args.gaussian_scale, "rr_crop_ratio_range": args.rr_crop_ratio_range, "output_size": args.output_size, "warps": args.warps, "radius": args.radius,
                "epsilon": args.epsilon, "magnitude_range": args.magnitude_range, "downsample_ratio": args.downsample_ratio, "to_crop_ratio_range": args.to_crop_ratio_range,
                "bw_cmax":0.1, "em_cmax":0.5, "pl_cmax":0.2, "bs_cmax":1}
    transformations = args.trafos
    checkpoint_config = os.path.join("checkpoints", args.config_name)
    config_file = checkpoint_config if args.resume and os.path.isfile(
        checkpoint_config) else args.config_name
    config = yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)
    args_dict = vars(args)
    for key in set(config.keys()).union(set(args_dict.keys())):
        config[key] = config[key] if (key not in args_dict.keys() or key in args_dict.keys(
        ) and key in config.keys() and args_dict[key] is None) else args_dict[key]
    if args.target_folders is not None:
        config["dataset"]["target_folders"] = args.target_folders
    config["dataset"]["percentage"] = args.percentage if args.percentage is not None else config["dataset"]["percentage"]
    config["dataset"]["filter_cinc"] = args.filter_cinc if args.filter_cinc is not None else config["dataset"]["filter_cinc"]
    if args.base_model != 'OneLayerLinear': # resnet flavors
        config["model"]["base_model"] = args.base_model if args.base_model is not None else config["model"]["base_model"]
        config["model"]["widen"] = args.widen if args.widen is not None else config["model"]["widen"]
    else: # OneLayerLinear
        config["model"]["input_size"] = np.product(eval(config["dataset"]["input_shape"]))
        config["model"]["activation"] = args.activation if args.activation is not None else "F.relu"
    config["optimizer_name"] = args.optimizer_name if args.optimizer_name is not None else "Adam" # "SGD"
    if args.out_dim is not None:
        config["model"]["out_dim"] = args.out_dim
    init_logger(config)
    dataset = SimCLRDataSetWrapper(
        config['batch_size'], **config['dataset'], transformations=transformations, t_params=t_params)
    for i, t in enumerate(dataset.transformations):
        logger.info(str(i) + ". Transformation: " +
                    str(t) + ": " + str(t.get_params()))
    date = time.asctime()
    label_to_num_classes = {"label_all": 71, "label_diag": 44, "label_form": 19,
                            "label_rhythm": 12, "label_diag_subclass": 23, "label_diag_superclass": 5}
    ptb_num_classes = label_to_num_classes[config["eval_dataset"]["ptb_xl_label"]]
    abr = {"Transpose": "Tr", "TimeOut": "TO", "DynamicTimeWarp": "DTW", "RandomResizedCrop": "RRC", "ChannelResize": "ChR", "GaussianNoise": "GN",
           "TimeWarp": "TW", "ToTensor": "TT", "GaussianBlur": "GB", "BaselineWander": "BlW", "PowerlineNoise": "PlN", "EMNoise": "EM", "BaselineShift": "BlS"}
    trs = re.sub(r"[,'\]\[]", "", str([abr[str(tr)] if abr[str(tr)] not in [
                 "TT", "Tr"] else '' for tr in dataset.transformations]))
    name = str(date) + "_" + method + "_" + str(
        time.time_ns())[-3:] + "_" + trs[1:]
    tb_logger = TensorBoardLogger(args.log_dir, name=name, version='', log_graph=True)
    config["log_dir"] = os.path.join(args.log_dir, name)
    print(config)
    return config, dataset, date, transformations, t_params, ptb_num_classes, tb_logger

def aftertrain_routine(config, args, trainer, pl_model, datamodule, callbacks):
    scores = {}
    for ca in callbacks:
        if isinstance(ca, SSLOnlineEvaluator):
            scores[str(ca)] = {"macro": ca.best_macro}

    results = {"config": config, "trafos": args.trafos, "scores": scores}

    with open(os.path.join(config["log_dir"], "results.pkl"), 'wb') as handle:
        pickle.dump(results, handle)

    trainer.save_checkpoint(os.path.join(config["log_dir"], "checkpoints", "model.ckpt"))
    with open(os.path.join(config["log_dir"], "config.txt"), "w") as text_file:
        print(config, file=text_file)

def cli_main():
    from pytorch_lightning import Trainer
    from online_evaluator import SSLOnlineEvaluator
    from ecg_datamodule import ECGDataModule
    from clinical_ts.create_logger import create_logger
    from os.path import exists
    
    parser = ArgumentParser()
    parser = parse_args(parser)
    logger.info("parse arguments")
    args = parser.parse_args()
    config, dataset, date, transformations, t_params, ptb_num_classes, tb_logger = pretrain_routine(args)

    # data
    ecg_datamodule = ECGDataModule(config, transformations, t_params, ptb_num_classes=ptb_num_classes)
    if args.base_model == 'OneLayerLinear':
        if args.output_size > 0 and 'RandomResizedCrop' in args.trafos:
            config["model"]["input_size"] = eval(config["dataset"]["input_shape"])[0] * args.output_size
        config["model"]["num_ftrs"] = ecg_datamodule.num_samples**4 / config["model"]["input_size"]**3
        config["model"]["num_ftrs"] *= args.ftr_multiplier
        config["model"]["num_ftrs"] = int(config["model"]["num_ftrs"])

    callbacks = []
    if args.early_stopping:
        callbacks.append(EarlyStopping(monitor="val/val_loss", mode="min"))
    if args.best_model_ckpt:
        callbacks.append(ModelCheckpoint(
            dirpath=os.path.join(config["log_dir"], "checkpoints"),
            monitor="val/val_loss", mode="min", save_top_k=1, save_last=True))
    if args.run_callbacks:
        # callback for online linear evaluation/fine-tuning
        linear_evaluator = SSLOnlineEvaluator(drop_p=0, z_dim=512, num_classes=ptb_num_classes,
            hidden_dim=None, lin_eval_epochs=config["eval_epochs"], eval_every=config["eval_every"],
            mode="linear_evaluation", verbose=True)

        # TODO(loganesian): wrap in a flag.
        # fine_tuner = SSLOnlineEvaluator(drop_p=0, z_dim=512, num_classes=ptb_num_classes,
        #     hidden_dim=None, lin_eval_epochs=config["eval_epochs"], eval_every=config["eval_every"],
        #     mode="fine_tuning", verbose=False)
   
        callbacks.append(linear_evaluator)
        # callbacks.append(fine_tuner) # TOOD(loganesian): wrap in a flag.

    trainer = Trainer(logger=tb_logger, max_epochs=config["epochs"], gpus=args.gpus, # distributed_backend=args.distributed_backend,
                      auto_lr_find=False, num_nodes=args.num_nodes, precision=config["precision"], callbacks=callbacks,
                      log_every_n_steps=1)

    # pytorch lightning module
    if args.base_model == "OneLayerLinear":
        # hardcoding 512 to match the dimensionality of the features learned with resnet model
        # the only parameter we will vary is the number of hidden units.
        model = OneLayerLinear(config["model"]["input_size"], config["model"]["num_ftrs"],
            512, activation=eval(config["model"]["activation"]))
    else:
        model = ResNetSimCLR(**config["model"])

    pl_model = CustomSimCLR(
            config["batch_size"], ecg_datamodule.num_samples, warmup_epochs=config["warm_up"], lr=config["lr"],
            out_dim=config["model"]["out_dim"], config=config, optimizer_name=config["optimizer_name"],
            transformations=ecg_datamodule.transformations, loss_temperature=config["loss"]["temperature"], weight_decay=eval(config["weight_decay"]))
    pl_model.encoder = model # Even though the model has a projection
    if args.base_model == 'OneLayerLinear' and args.force_linear_projection:
        pl_model.projection = Projection(
            input_dim=model.l1.in_features, hidden_dim=None, output_dim=config["model"]["out_dim"])
    else:
        pl_model.projection = Projection(
            input_dim=model.l1.in_features, hidden_dim=512, output_dim=config["model"]["out_dim"])

    # load checkpoint
    if args.checkpoint_path != "":
        if exists(args.checkpoint_path):
            logger.info("Retrieve checkpoint from " + args.checkpoint_path)
            pl_model.load_from_checkpoint(args.checkpoint_path)
        else:
            raise("checkpoint does not exist")

    # start training
    trainer.fit(pl_model, datamodule=ecg_datamodule)

    aftertrain_routine(config, args, trainer, pl_model, ecg_datamodule, callbacks)

if __name__ == "__main__":  
    cli_main()

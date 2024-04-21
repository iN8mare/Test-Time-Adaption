from __future__ import print_function, division

import torch.optim
from scipy import stats
import torchvision.transforms as T
import copy
from torchvision import models
import os
from torch.utils.data.dataloader import default_collate
import cv2
import math
from warnings import warn
from torch.autograd import Variable

import argparse
import data_loader
from SSHead import *
from rotation import *
from util import *
from tqdm import tqdm
from scipy.stats import spearmanr

os.environ["CUDA_VISIBLE_DEVICES"]="0"

import warnings
warnings.filterwarnings("ignore")
import random
use_gpu = True
Image.LOAD_TRUNCATED_IMAGES = True

from sam import SAM
from depth_anything.dpt import DepthAnything

from albumentations.augmentations.utils import MAX_VALUES_BY_DTYPE

class BaselineModel1(nn.Module):
    def __init__(self, num_classes, keep_probability, inputsize):

        super(BaselineModel1, self).__init__()
        self.fc1 = nn.Linear(inputsize, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.drop_prob = (1 - keep_probability)
        self.relu1 = nn.PReLU()
        self.drop1 = nn.Dropout(self.drop_prob)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.relu2 = nn.PReLU()
        self.drop2 = nn.Dropout(p=self.drop_prob)
        self.fc3 = nn.Linear(512, num_classes)
        self.sig = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Weight initialization reference: https://arxiv.org/abs/1502.01852
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            # elif isinstance(m, nn.Linear):
            #     m.weight.data.normal_(0, 0.02)
            #     m.bias.data.zero_()

    def forward(self, x):
        """
        Feed-forward pass.
        :param x: Input tensor
        : return: Output tensor
        """
        out = self.fc1(x)

        out = self.bn1(out)
        out = self.relu1(out)
        out = self.drop1(out)
        out = self.fc2(out)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.drop2(out)
        out = self.fc3(out)
        out = self.sig(out)
        # out_a = torch.cat((out_a, out_p), 1)

        # out_a = self.sig(out)
        return out

class Net(nn.Module):
    def __init__(self , resnet, net):
        super(Net, self).__init__()
        self.resnet_layer = resnet
        self.net = net


    def forward(self, x):
        x = self.resnet_layer(x)
        x = self.net(x)

        return x


def computeSpearman(test_data, model):
    ratings = []
    predictions = []
    with torch.no_grad():
        cum_loss = 0
        # for batch_idx, data in enumerate(dataloader_valid):
        for data, labels in tqdm(test_data):
            # inputs = data['image']
            inputs = data
            if use_gpu:
                try:
                    inputs, labels = Variable(inputs.float().cuda()), Variable(labels.float().cuda())
                except:
                    print(inputs, labels)
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            outputs_a = model(inputs)
            ratings.append(labels.float().cpu())
            predictions.append(outputs_a.float().cpu())

    ratings_i = np.vstack(ratings)
    predictions_i = np.vstack(predictions)
    a = ratings_i[:,0]
    b = predictions_i[:,0]
    sp = spearmanr(a, b)
    return sp


def exp_lr_scheduler(optimizer, epoch, lr_decay_epoch=2):
    """Decay learning rate by a factor of DECAY_WEIGHT every lr_decay_epoch epochs."""

    decay_rate =  0.9**(epoch // lr_decay_epoch)
    if epoch % lr_decay_epoch == 0:
        print('decay_rate is set to {}'.format(decay_rate))

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate

    return optimizer

def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


def to_float(img: np.ndarray, max_value = None) -> np.ndarray:
    if max_value is None:
        if img.dtype not in MAX_VALUES_BY_DTYPE:
            raise RuntimeError(f"Unsupported dtype {img.dtype}. Specify 'max_value' manually.")
        max_value = MAX_VALUES_BY_DTYPE[img.dtype]

    return (img / max_value).astype(np.float32)


def from_float(img: np.ndarray, dtype: np.dtype, max_value = None) -> np.ndarray:
    if max_value is None:
        if dtype not in MAX_VALUES_BY_DTYPE:
            msg = (
                f"Can't infer the maximum value for dtype {dtype}. "
                "You need to specify the maximum value manually by passing the max_value argument."
            )
            raise RuntimeError(msg)
        max_value = MAX_VALUES_BY_DTYPE[dtype]
    return (img * max_value).astype(dtype)


def image_compression(img: np.ndarray, quality: int, image_type=".jpg"):
    img = img.cpu().numpy().transpose(1, 2, 0)

    if image_type in [".jpeg", ".jpg"]:
        quality_flag = cv2.IMWRITE_JPEG_QUALITY
    elif image_type == ".webp":
        quality_flag = cv2.IMWRITE_WEBP_QUALITY
    else:
        NotImplementedError("Only '.jpg' and '.webp' compression transforms are implemented. ")

    input_dtype = img.dtype
    needs_float = False

    if input_dtype == np.float32:
        warn(
            "Image compression augmentation "
            "is most effective with uint8 inputs, "
            f"{input_dtype} is used as input.",
            UserWarning,
        )
        img = from_float(img, dtype=np.dtype("uint8"))
        needs_float = True
    elif input_dtype not in (np.uint8, np.float32):
        raise ValueError(f"Unexpected dtype {input_dtype} for image augmentation")

    _, encoded_img = cv2.imencode(image_type, img, (int(quality_flag), quality))
    img = cv2.imdecode(encoded_img, cv2.IMREAD_UNCHANGED)

    if needs_float:
        img = to_float(img, max_value=255)
    img = img.transpose(2, 0, 1)
    return torch.from_numpy(img)


class METAIQASolver(object):
    """Solver for training and testing hyperIQA"""
    def __init__(self, config, path, train_idx, test_idx):

        self.test_patch_num = config.test_patch_num

        net1 = models.resnet18(pretrained=True)
        net2 = BaselineModel1(1, 0.5, 1000)
        self.model = Net(resnet=net1, net=net2).cuda()

        self.head = head_on_layer2(config).cuda()

        self.ext = self.model.resnet_layer.cuda()
        self.ssh = ExtractorHead(self.ext, self.head).cuda()

        self.l1_loss = torch.nn.L1Loss().cuda()

        self.lr = config.lr

        base_optimizer = torch.optim.Adam

        if not config.fix_ssh:
            params_to_optimize = self.ssh.parameters()
        else:
            params_to_optimize = self.ext.parameters()

        if config.optimizer == "sam":
            self.optimizer_ssh = SAM(params_to_optimize, base_optimizer, lr=self.lr)
        elif config.optimizer == "adam":
            self.optimizer_ssh = torch.optim.Adam(params_to_optimize, lr=self.lr)
        else:
            raise NotImplementedError("only 'adam' and 'sam' else include in ur code")

        train_loader = data_loader.DataLoader(config, config.dataset, path,
                                              train_idx, config.patch_size,
                                              config.train_patch_num,
                                              batch_size=config.batch_size, istrain=True)

        test_loader = data_loader.DataLoader(config, config.dataset, path,
                                             test_idx, config.patch_size,
                                             config.test_patch_num, batch_size=config.batch_size, istrain=True)

        self.train_data = train_loader.get_data()
        self.test_data = test_loader.get_data()
        self.rank_loss = nn.BCELoss()

    def test(self,svPath,seed,pretrained=False):
        """Testing"""
        self.model.train(False)

        if pretrained:
            self.model.load_state_dict(torch.load(svPath+'model_IQA/TID2013_KADID10K_IQA_Meta_resnet18_38'))

        pred_scores = []
        gt_scores = []

        with torch.no_grad():
            steps2 = 0
            for img, label in tqdm(self.test_data):

                img=img['image']

                img = torch.tensor(img.cuda())
                label = torch.tensor(label.cuda())
                pred = self.model(img)

                pred_scores = pred_scores + pred.cpu().tolist()
                gt_scores = gt_scores + label.cpu().tolist()
                steps2 += 1
                if steps2 % 50 == 0:

                    pred_scores4 = np.mean(np.reshape(np.array(pred_scores), (-1, 1)), axis=1)
                    gt_scores4 = np.mean(np.reshape(np.array(gt_scores), (-1, 1)), axis=1)

                    test_srcc4, _ = stats.spearmanr(pred_scores4, gt_scores4)
                    test_plcc4, _ = stats.pearsonr(pred_scores4, gt_scores4)
                    print('After {} images test_srcc : {} \n test_plcc:{}'.format(steps2, test_srcc4, test_plcc4))

        pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, 1)), axis=1)
        gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, 1)), axis=1)
        test_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
        test_plcc, _ = stats.pearsonr(pred_scores, gt_scores)

        return test_srcc, test_plcc

    def adapt(self, data_dict, config, old_net, depth_anything):
        inputs = data_dict['image']
        f_low = []
        f_high = []
        old_net.eval()

        with torch.no_grad():
            pred0 = old_net(data_dict['image'].cuda())

            if config.rank:
                sigma1 = 40 + np.random.random()*20
                sigma2 = 5 + np.random.random()*15

                data_dict['blur_high'] = T.GaussianBlur(kernel_size=(5, 5), sigma=(sigma1))(inputs).cuda()
                data_dict['blur_low'] = T.GaussianBlur(kernel_size=(5, 5), sigma=(sigma2))(inputs).cuda()

                id_dict = {0: data_dict['comp_high'], 1: data_dict['comp_low'], 2: data_dict['nos_high'],
                           3: data_dict['nos_low'], 4: data_dict['blur_high'], 5: data_dict['blur_low']}

                pred1= old_net(data_dict['comp_high'].cuda())

                pred2 = old_net(data_dict['comp_low'].cuda())

                pred3= old_net(data_dict['nos_high'].float().cuda())

                pred4= old_net(data_dict['nos_low'].float().cuda())

                pred5= old_net(data_dict['blur_high'].cuda())

                pred6= old_net(data_dict['blur_low'].cuda())

                try:
                    comp = torch.unsqueeze(torch.abs(pred2 - pred1), dim=1)  # - torch.abs(pred0 - pred2), dim=1)
                except:
                    comp = (torch.ones(1, 1) * (torch.abs(pred2 - pred1)).item()).cuda()

                try:
                    nos = torch.unsqueeze(torch.abs(pred4 - pred3), dim=1)
                except:
                    nos = (torch.ones(1, 1) * (torch.abs(pred4 - pred3)).item()).cuda()

                try:
                    blur = torch.unsqueeze(torch.abs(pred6 - pred5), dim=1)
                except:
                    blur = (torch.ones(1, 1) * (torch.abs(pred6 - pred5)).item()).cuda()

                all_diff = torch.cat([comp, nos, blur], dim=1)

                for p in range(len(pred0)):
                    if all_diff[p].argmax().item() == 0:
                        f_low.append(id_dict[1][p].cuda())
                        f_high.append(id_dict[0][p].cuda())
                        # print('comp', end=" ")
                    if all_diff[p].argmax().item() == 1:
                        f_low.append(id_dict[3][p].cuda())
                        f_high.append(id_dict[2][p].cuda())
                        # print('nos', end=" ")
                    if all_diff[p].argmax().item() == 2:
                        f_low.append(id_dict[5][p].cuda())
                        f_high.append(id_dict[4][p].cuda())
                        # print('blur', end=" ")

                f_low = torch.squeeze(torch.stack(f_low), dim=1)
                f_high = torch.squeeze(torch.stack(f_high), dim=1)

        if config.comp:
            f_low = data_dict['comp_low'].cuda()
            f_high = data_dict['comp_high'].cuda()
        if config.nos:
            f_low = data_dict['nos_low'].cuda()
            f_high = data_dict['nos_high'].cuda()
        if config.blur:
            sigma1 = 40 + np.random.random() * 20
            sigma2 = 5 + np.random.random() * 15

            f_low = T.GaussianBlur(kernel_size=(5, 5), sigma=(sigma2))(inputs).cuda()
            f_high = T.GaussianBlur(kernel_size=(5, 5), sigma=(sigma1))(inputs).cuda()

        if config.contrastive:
            f_low = data_dict['image1'].cuda()
            f_high = data_dict['image2'].cuda()

        m = nn.Sigmoid()

        for param in self.ssh.parameters():
            param.requires_grad = False

        for layer in self.ssh.ext.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.requires_grad_(True)

        if config.fix_ssh:
            self.ssh.eval()
            self.ssh.ext.train()
        else:
            self.ssh.train()

        loss_hist = []
        cur_loss = {'rank_loss': None, 'grp_contr_loss': None, 'depth_loss': None}

        for iteration in range(config.niter):
            target = torch.ones(inputs.shape[0]).cuda()

            for pass_num in range(2):  # for first and second pass of SAM optimizer
                if config.optimizer == 'adam' and pass_num == 1:
                    continue   # only 1 pass in case of adam / sgd

                total_loss = 0
                if config.rank or config.blur or config.comp or config.nos:

                    f_low_feat = self.ssh(f_low)
                    f_high_feat = self.ssh(f_high)
                    f_actual = self.ssh(inputs.cuda())

                    dist_high = torch.nn.PairwiseDistance(p=2)(f_high_feat, f_actual)
                    dist_low = torch.nn.PairwiseDistance(p=2)(f_low_feat, f_actual)
                    loss = self.rank_loss(m(dist_high - dist_low), target) * config.rank_w
                    if iteration == 0 and pass_num == 0:
                        # print("rank loss: ", loss.item())
                        cur_loss['rank_loss'] = loss.item()
                    total_loss += loss

                if config.contrastive:
                    f_neg_feat = self.ssh(f_low)
                    f_pos_feat = self.ssh(f_high)

                    loss_fn = ContrastiveLoss(f_pos_feat.shape[0], 1).cuda()

                    loss = loss_fn(f_neg_feat, f_pos_feat)
                    total_loss += loss

                if config.depth:
                    #  --------------------- DEPTH ANYTHING ----------------------#
                    depth_org, depth_80, depth_60, depth_40 = [], [], [], []
                    for i in range(inputs.shape[0]):
                        compressed_80 = image_compression(inputs[i], 80).unsqueeze(dim=0)
                        compressed_60 = image_compression(inputs[i], 60).unsqueeze(dim=0)
                        compressed_40 = image_compression(inputs[i], 40).unsqueeze(dim=0)
                        compressed_tensors = torch.cat(
                            [inputs[i].unsqueeze(dim=0), compressed_80, compressed_60, compressed_40])

                        with torch.no_grad():
                            depth = depth_anything(compressed_tensors.cuda())

                        for i in range(depth.shape[0]):
                            cur_depth = (depth[i] - depth[i].min()) / (depth[i].max() - depth[i].min()) * 255.0
                            cur_depth = cur_depth.cpu().numpy().astype(np.uint8)
                            cur_depth = cv2.applyColorMap(cur_depth, cv2.COLORMAP_INFERNO)
                            cur_depth = cur_depth.transpose(2, 1, 0)
                            cur_depth = torch.from_numpy(cur_depth).type(torch.float32)
                            cur_depth = torch.nn.functional.normalize(cur_depth, p=2.0, dim=0)
                            if i == 0:
                                depth_org.append(cur_depth)
                            elif i == 1:
                                depth_80.append(cur_depth)
                            elif i == 2:
                                depth_60.append(cur_depth)
                            elif i == 3:
                                depth_40.append(cur_depth)
                            else:
                                raise ValueError("i to be in range [0,3]")

                    depth_org = torch.from_numpy(np.array(depth_org))
                    depth_80 = torch.from_numpy(np.array(depth_80))
                    depth_60 = torch.from_numpy(np.array(depth_60))
                    depth_40 = torch.from_numpy(np.array(depth_40))

                    emb_org = old_net(depth_org.cuda())
                    emb_80 = old_net(depth_80.cuda())
                    emb_60 = old_net(depth_60.cuda())
                    emb_40 = old_net(depth_40.cuda())

                    depth_loss = 0
                    for i in range(emb_org.shape[0]):
                        depth_loss += (torch.mean(emb_org[i] - emb_80) ** 2 + torch.mean(
                            emb_org[i] - emb_60) ** 2 + torch.mean(emb_org[i] - emb_40) ** 2) / 3

                    depth_loss /= emb_org.shape[0]
                    depth_loss *= config.depth_w

                    total_loss += depth_loss
                    if iteration == 0 and pass_num == 0:
                        #print("depth loss: ", depth_loss.item())
                        cur_loss['depth_loss'] = depth_loss.item()

                if config.group_contrastive:

                    idx = np.argsort(pred0.cpu(), axis=0)

                    f_feat = self.ssh(inputs.cuda())

                    f_pos_feat = []
                    f_neg_feat = []

                    for n in range(len(f_feat) // 4):
                        f_pos_feat.append(f_feat[idx[n]])
                        f_neg_feat.append(f_feat[idx[-n - 1]])

                    f_pos_feat = torch.squeeze(torch.stack(f_pos_feat), dim=1)
                    f_neg_feat = torch.squeeze(torch.stack(f_neg_feat), dim=1)

                    loss_fn = GroupContrastiveLoss(f_pos_feat.shape[0], 1).cuda()
                    group_contrastive_loss = loss_fn(f_neg_feat, f_pos_feat) * config.contr_w

                    total_loss += group_contrastive_loss
                    if iteration == 0 and pass_num == 0:
                        #print("grp contrastive loss: ", group_contrastive_loss.item())
                        cur_loss['grp_contr_loss'] = group_contrastive_loss.item()

                if config.rotation:
                    inputs_ssh, labels_ssh = rotate_batch(inputs.cuda(), 'rand')
                    outputs_ssh = self.ssh(inputs_ssh.float())
                    loss = nn.CrossEntropyLoss()(outputs_ssh, labels_ssh.cuda())
                    total_loss += loss

                if pass_num == 0:
                    # loss = loss_function(output, model(input))  # use this loss for any training statistics
                    total_loss.backward()
                    if config.optimizer == 'sam':
                        self.optimizer_ssh.first_step(zero_grad=True)
                    else:
                        self.optimizer_ssh.step()
                    loss_hist.append(total_loss.detach().cpu())  # appending here since only this loss is to be used

                if pass_num == 1:
                    total_loss.backward()  # make sure to do a full forward pass
                    self.optimizer_ssh.second_step(zero_grad=True)

                # loss.backward()
                # self.optimizer_ssh.step()
        print(cur_loss)
        return loss_hist

    def new_ttt(self, svPath, config):

        if config.online:
            self.model.load_state_dict(torch.load(svPath+'model_IQA/TID2013_KADID10K_IQA_Meta_resnet18_38'))

        steps = 0
        pred_scores_old = []
        pred_scores = []
        gt_scores = []
        mse_all = []
        mse_all_old = []

        pbar = tqdm(self.test_data, leave=False)

        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format("vitl")).to(
            DEVICE).eval()
        total_params = sum(param.numel() for param in depth_anything.parameters())
        print('Total parameters: {:.2f}M'.format(total_params / 1e6))

        for data_dict, label in pbar:
            img = data_dict['image']

            if not config.online:
                self.model.load_state_dict(torch.load(svPath+'model_IQA/TID2013_KADID10K_IQA_Meta_resnet18_38'))

            label = torch.as_tensor(label.cuda()).requires_grad_(False)

            old_net = copy.deepcopy(self.model)
            old_net.load_state_dict(torch.load(svPath+'model_IQA/TID2013_KADID10K_IQA_Meta_resnet18_38'))

            if config.group_contrastive:
                if len(img) > 3:
                    loss_hist = self.adapt(data_dict, config, old_net, depth_anything)
                elif config.rank or config.blur or config.comp or config.nos or config.contrastive or config.rotation:
                    config.group_contrastive=False
                    loss_hist = self.adapt(data_dict, config, old_net, depth_anything)
            elif config.rank or config.blur or config.comp or config.nos or config.contrastive or config.rotation:
                loss_hist = self.adapt(data_dict, config, old_net, depth_anything)

            with open(f"{config.path_to_save_loss}/{config.contr_w}_{config.rank_w}_{config.depth_w}_losses.txt", "a") as f:
                for i in range(len(loss_hist)):
                    f.write(str(loss_hist[i].item()) + "\n")

            old_net.load_state_dict(torch.load(svPath+'model_IQA/TID2013_KADID10K_IQA_Meta_resnet18_38'))

            self.model.eval()
            old_net.eval()

            mse, pred = self.test_single_iqa(self.model.cuda(), img.cuda(), label.cuda())
            mse_old, pred_old = self.test_single_iqa(old_net.cuda(), img.cuda(), label.cuda())

            self.model.train()

            pred_scores = pred_scores + pred.cpu().tolist()
            pred_scores_old = pred_scores_old + pred_old.cpu().tolist()
            gt_scores = gt_scores + label.cpu().tolist()

            mse_all.append(mse.cpu())
            mse_all_old.append(mse_old.cpu())
            steps += 1

            if steps % 20 == 0:
                pred_scores1 = np.mean(np.reshape(np.array(pred_scores), (-1, self.test_patch_num)), axis=1)
                pred_scores_old1 = np.mean(np.reshape(np.array(pred_scores_old), (-1, self.test_patch_num)), axis=1)
                gt_scores1 = np.mean(np.reshape(np.array(gt_scores), (-1, self.test_patch_num)), axis=1)

                test_srcc_old, _ = stats.spearmanr(pred_scores_old1, gt_scores1)
                test_plcc_old, _ = stats.pearsonr(pred_scores_old1, gt_scores1)

                test_srcc, _ = stats.spearmanr(pred_scores1, gt_scores1)
                test_plcc, _ = stats.pearsonr(pred_scores1, gt_scores1)

                print('After {} images test_srcc old : {}  new {} \n test_plcc old:{} new:{}'.format(steps, test_srcc_old,
                                                                                                   test_srcc,
                                                                                                   test_plcc_old,
                                                                                                   test_plcc))

                with open(f"{config.path_to_save_loss}/{config.contr_w}_{config.rank_w}_{config.depth_w}.txt", "a") as f:
                    f.write('After {} images test_srcc old : {}  new {} \n test_plcc old:{} new:{} \n\n'.format(steps, test_srcc_old,
                                                                                                   test_srcc,
                                                                                                   test_plcc_old,
                                                                                                   test_plcc))

        pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, self.test_patch_num)), axis=1)
        pred_scores_old = np.mean(np.reshape(np.array(pred_scores_old), (-1, self.test_patch_num)), axis=1)
        gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, self.test_patch_num)), axis=1)

        test_srcc_old, _ = stats.spearmanr(pred_scores_old, gt_scores)
        test_plcc_old, _ = stats.pearsonr(pred_scores_old, gt_scores)

        test_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
        test_plcc, _ = stats.pearsonr(pred_scores, gt_scores)


        with open(f"{config.path_to_save_loss}/{config.contr_w}_{config.rank_w}_{config.depth_w}.txt", "a") as f:
            f.write('srcc_computed {}, plcc_computed {} \n'.format(test_srcc, test_plcc))
            f.write('srcc_computed_old {}, plcc_computed_old {} \n'.format(test_srcc_old, test_plcc_old))

        return test_srcc_old, test_plcc_old, test_srcc, test_plcc

    def test_single_iqa(self, model, image, label):
        model.eval()
        with torch.no_grad():
            pred = model(image)
            mse_loss = self.l1_loss(label, pred)
        model.train()
        return mse_loss, pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default='koniq',
                        help='Support datasets: livec|koniq-10k|bid|live|csiq|tid2013')
    parser.add_argument('--train_patch_num', dest='train_patch_num', type=int, default=1,
                        help='Number of sample patches from training image')
    parser.add_argument('--test_patch_num', dest='test_patch_num', type=int, default=1,
                        help='Number of sample patches from testing image')
    parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight', dest='weight', type=float, default=1, help='Weight')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--patch_size', dest='patch_size', type=int, default=224,
                        help='Crop size for training & testing image patches')
    parser.add_argument('--seed', dest='seed', type=int, default=2021, help='for reproducing the results')
    parser.add_argument('--svpath', dest='svpath', type=str, default='./results/',
                        help='the path to save the info')
    parser.add_argument('--datapath', default='')
    parser.add_argument('--contrastive', action='store_true')
    parser.add_argument('--group_contrastive', action='store_true')
    parser.add_argument('--rank', action='store_true')
    parser.add_argument('--depth', action='store_true')
    parser.add_argument('--comp', action='store_true')
    parser.add_argument('--nos', action='store_true')
    parser.add_argument('--blur', action='store_true')
    parser.add_argument('--rotation', action='store_true')
    parser.add_argument('--contrique', action='store_true')
    parser.add_argument('--online', action='store_true')
    parser.add_argument('--fix_ssh', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--niter', default=3, type=int)
    parser.add_argument('--run', dest='run', type=int, default=1, help='for running at multiple seeds')
    parser.add_argument('--contr_w', type=int, default=1, help='grp_contr_w')
    parser.add_argument('--rank_w', type=int, default=1, help='rank loss w')
    parser.add_argument('--depth_w', type=int, default=100, help='depth loss w')
    parser.add_argument('--optimizer', type=str, default="sam", help="optimizer -> 'adam' or 'sam' ")

    config = parser.parse_args()
    config.datapath = config.dataset
    if config.rank is False:
        config.rank_w = 0
    elif config.rank_w == 0:
        config.rank = False

    if config.group_contrastive is False:
        config.contr_w = 0
    elif config.contr_w == 0:
        config.group_contrastive = False

    if config.depth is False:
        config.depth_w = 0
    elif config.depth_w == 0:
        config.depth = False

    print("Optimizing the following --> ")
    print(f"grp_contr_w: {config.contr_w} | rank_w: {config.rank_w} | depth_w: {config.depth_w}")
    print(f"group_contrastive: {config.group_contrastive} | rank: {config.rank} | depth: {config.depth}")

    config.path_to_save_loss = config.svpath + config.dataset
    os.makedirs(config.path_to_save_loss, exist_ok=True)

    print("results will be saved at: ", config.path_to_save_loss)
    print(f"using {config.optimizer} optimizer")

    if os.path.exists(
            f"{config.path_to_save_loss}/{config.contr_w}_{config.rank_w}_{config.depth_w}.txt"):
        raise Exception(
            f"{config.path_to_save_loss}/{config.contr_w}_{config.rank_w}_{config.depth_w}.txt file exists! Delete previous file to avoid mixing new results with previous ones")

    if os.path.exists(f"{config.path_to_save_loss}/{config.contr_w}_{config.rank_w}_{config.depth_w}_losses.txt"):
        raise Exception(
            f"{config.path_to_save_loss}/{config.contr_w}_{config.rank_w}_{config.depth_w}_losses.txt file exists! Delete previous file to avoid mixing new results with previous ones")

    folder_path = {
        'live': config.datapath,
        'csiq': config.datapath,
        'tid2013': config.datapath,
        'kadid10k': config.datapath,
        'clive': config.datapath,
        'koniq': config.datapath,
        'fblive': config.datapath,
        'pipal': config.datapath,
        'cidiq': config.datapath,
        'dslr': config.datapath
    }

    img_num = {
        'live': list(range(0, 29)),
        'csiq': list(range(0, 30)),
        'kadid10k': list(range(0, 80)),
        'tid2013': list(range(0, 25)),
        'clive': list(range(0, 1162)),
        'koniq': list(range(0, 10073)),
        'fblive': list(range(0, 39810)),
        'pipal': list(range(0, 5800)),
        'cidiq': list(range(0, 475)),
    'dslr': list(range(0, 1035))
    }

    svPath = config.svpath
    os.makedirs(svPath, exist_ok=True)

    rho_s_list, rho_p_list = [], []

    for mul in range(config.run):

        # fix the seed if needed for reproducibility
        if config.seed == 0:
            pass
        else:
            if mul != 0:
                config.seed = config.seed + np.random.randint(1000)
            print('we are using the seed = {}'.format(config.seed))
            torch.manual_seed(config.seed)
            torch.cuda.manual_seed(config.seed)
            np.random.seed(config.seed)
            random.seed(config.seed)

        sel_num = img_num[config.dataset]

        # Randomly select 80% images for training and the rest for testing
        random.shuffle(sel_num)
        train_index = sel_num[0:int(round(0.8 * len(sel_num)))]
        # test_index = sel_num[int(round(0.8 * len(sel_num))):len(sel_num)]
        test_index = sel_num

        solver = METAIQASolver(config, folder_path[config.dataset], train_index, test_index)

        if config.test:
            srcc_computed1, plcc_computed1 = solver.test(config.svpath, config.seed, pretrained=True)
            print('srcc_computed_test {}, plcc_computed_test {}'.format(srcc_computed1, plcc_computed1))
            continue
        else:
            test_srcc_old, test_plcc_old, srcc_computed, plcc_computed = solver.new_ttt(config.svpath, config)
            print('srcc_computed {}, plcc_computed {}'.format(srcc_computed, plcc_computed))
            print('srcc_computed_old {}, plcc_computed_old {}'.format(test_srcc_old, test_plcc_old))
            rho_s_list.append(srcc_computed)
            rho_p_list.append(plcc_computed)

    if not config.test:
        final_rho_s = np.mean(np.array(rho_s_list))
        final_rho_p = np.mean(np.array(rho_p_list))

        print(' final_srcc new {} \n final_plcc new:{}'.format(final_rho_s, final_rho_p))


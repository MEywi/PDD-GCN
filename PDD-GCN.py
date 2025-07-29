import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from layers.squeeze_embedding import SqueezeEmbedding
from layers.point_wise_feed_forward import PositionwiseFeedForward
from transformers import AutoModel, AutoTokenizer,AutoConfig
import math
from item import pgd_attack_with_cross_entropy
class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 32)  
        self.conv2 = GCNConv(32, num_classes)   
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
 
                x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        
               x = self.conv2(x, edge_index)
        
        return F.log_softmax(x, dim=1)

class TripletLoss(torch.nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.pdist = PairwiseDistance(2)
    def forward(self, anchor, positive, negative):
        pos_sim = F.cosine_similarity(anchor, positive)
        neg_sim = F.cosine_similarity(anchor, negative)
        pos_dist = 1 - pos_sim
        neg_dist   = 1 - neg_sim
        tmp =pos_dist - neg_dist
        hinge_dist = torch.clamp(self.margin + pos_dist - neg_dist, min=0.0)
        loss = torch.mean(hinge_dist)
        return loss

class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class PDD-GCN(nn.Module):
    def __init__(self, bert, tokenizer, opt):
        super(PDD-GCN, self).__init__()
        self.opt = opt
        self.bert = bert
        self.squeeze_embedding_False = SqueezeEmbedding(batch_first=False)
        self.squeeze_embedding_True = SqueezeEmbedding(batch_first=True)
        self.dropout = nn.Dropout(opt.dropout)
        self.layernorm = LayerNorm(opt.bert_dim)
        self.proj_drop = nn.Dropout(opt.proj_drop)
        self.batch_size = opt.batch_size_train
        config = AutoConfig.from_pretrained("../bert_unscaed")
        self.bert = AutoModel.from_pretrained('../bert_unscaed',config=config)
        self.dropout_zheng = nn.Dropout(opt.zheng_drop)
        self.x_liner = nn.Linear(3, 768)
        self.fc = nn.Linear(768, 3)
        self.fc_aspect_other_one = nn.Linear(768, 3)
        self.fc_aspect_other_two = nn.Linear(768, 3)
        self.triplet_loss = TripletLoss(margin=)

    def project_tensor(self, a, b):
        proj_B_on_A = torch.zeros_like(a).to(self.opt.device)
        for i in range(a.shape[0]):
            tmp_a = a[i]
            tmp_b = b[i]

            dot_product_a_b = torch.dot(tmp_a, tmp_b)
            v_length = torch.norm(b[i], keepdim=True)  
            proj_B_on_A[i] = (dot_product_a_b / (v_length ** 2)) * b[i]
            proj_B_on_A[i] = self.proj_drop(F.relu(proj_B_on_A[i]))
        return proj_B_on_A
              
    def forward(self, inputs):
        sentence = inputs[0].to(self.opt.device)
        sentence_last_hidden_state = sentence.last_hidden_state
        sentence_pooler_output = self.layernorm(sentence.pooler_output)
        sentence_pooler_output = self.dropout(F.relu(self.layernorm(sentence_pooler_output)))
        sentence_last_hidden_state = self.dropout(F.relu(self.layernorm(sentence_last_hidden_state)))
        aspect_sen = inputs[2].to(self.opt.device)
        aspect_one_sentence = self.bert(inputs[4].to(self.opt.device)).last_hidden_state
        original_vector_uni_one_drop = self.dropout(F.relu(self.layernorm(sentence_pooler_output)))
        sentence_last_hidden_state = sentence_last_hidden_state.to(self.opt.device)
        aspect_one_sentence = aspect_one_sentence.to(self.opt.device)
        N = torch.cat((sentence_last_hidden_state,aspect_one_sentence),dim = 0)
        N = F.avg_pool1d(N.transpose(1, 2), kernel_size=60).transpose(1, 2).squeeze(1)
        original_vector = N[:sentence_last_hidden_state.shape[0],:]
        aspect_one_sentence=N[aspect_one_sentence.shape[0]:,:]
        original_vector_uni_one = AEN_BERT.project_tensor(self, aspect_one_sentence,original_vector)
        target_label = inputs[1].to(self.opt.device)
        perturbed_vector = pgd_attack_with_cross_entropy(original_vector_uni_one, target_label,inputs[5])
        contrast_learn = ContrastiveLoss(original_vector_uni_one.shape[0])
        contrast_loss = contrast_learn(original_vector_uni_one,original_vector_uni_one_drop,perturbed_vector)
        x = original_vector_uni_one
        x = F.relu(x)
        x = self.fc(x)
        torch_x = torch.tensor(x)
        torch_x = self.x_liner(torch_x)
        loss_tri = self.triplet_loss.forward(torch_x, original_vector_uni_one, aspect_one_sentence)
        return x, contrast_loss, loss_tri, original_vector_uni_one,perturbed_vector




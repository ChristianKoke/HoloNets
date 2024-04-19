import torch
from torch import nn, optim
import pytorch_lightning as pl
from torch_sparse import SparseTensor
import torch.nn.functional as F
from torch.nn import ModuleList, Linear
from torch_geometric.nn import (GCNConv,
    JumpingKnowledge)

from src.datasets.data_utils import get_norm_adj


def get_conv(conv_type, input_dim, output_dim, alpha, K_plus = 3, K_minus = 1, zero_order = False, exponent = -0.5, weight_penalty = 'exp'):
    if conv_type == "gcn":
        return GCNConv(input_dim, output_dim, add_self_loops=False)
    elif conv_type == "fabernet":
        return FaberConv(input_dim, output_dim, alpha=alpha, K_plus = K_plus, exponent = exponent, weight_penalty = weight_penalty, zero_order = zero_order)
    elif conv_type == "complex-fabernet":
        return ComplexFaberConv(input_dim, output_dim, alpha=alpha, K_plus = K_plus, exponent = exponent, weight_penalty = weight_penalty, zero_order = zero_order)
    else:
        raise ValueError(f"Convolution type {conv_type} not supported")




class FaberConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, alpha, K_plus=1, exponent = -0.25, weight_penalty = 'exp', zero_order = False):
        super(FaberConv, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.K_plus = K_plus
        self.exponent = exponent
        self.weight_penalty = weight_penalty
        self.zero_order = zero_order

        if self.zero_order:
            #Zero Order Lins
            #Source to destination 
            self.lin_src_to_dst_zero = Linear(input_dim, output_dim)
            #Source to destination 
            self.lin_dst_to_src_zero = Linear(input_dim, output_dim)



        #Lins for positive powers:
        self.lins_src_to_dst = torch.nn.ModuleList([
            Linear(input_dim, output_dim) for _ in range(K_plus)
        ])

        self.lins_dst_to_src = torch.nn.ModuleList([
            Linear(input_dim, output_dim) for _ in range(K_plus)
        ])

        self.alpha = alpha
        self.adj_norm, self.adj_t_norm = None, None

    def forward(self, x, edge_index):
        if self.adj_norm is None:
            row, col = edge_index
            num_nodes = x.shape[0]

            

            adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
            self.adj_norm = get_norm_adj(adj, norm="dir", exponent = self.exponent)

            adj_t = SparseTensor(row=col, col=row, sparse_sizes=(num_nodes, num_nodes))
            self.adj_t_norm = get_norm_adj(adj_t, norm="dir", exponent = self.exponent)

        y   = self.adj_norm   @ x
        y_t = self.adj_t_norm @ x
        sum_src_to_dst = self.lins_src_to_dst[0](y) 
        sum_dst_to_src = self.lins_dst_to_src[0](y_t) 
        if self.zero_order:
            sum_src_to_dst =  sum_src_to_dst + self.lin_src_to_dst_zero(x)
            sum_dst_to_src =  sum_dst_to_src + self.lin_dst_to_src_zero(x)




        if self.K_plus > 1:
            if self.weight_penalty == 'exp':
                for i in range(1,self.K_plus):
                    y   = self.adj_norm   @ y
                    y_t = self.adj_t_norm @ y

                    sum_src_to_dst = sum_src_to_dst + self.lins_src_to_dst[i](y)/(2**i)
                    sum_dst_to_src = sum_dst_to_src + self.lins_dst_to_src[i](y_t)/(2**i)

            elif self.weight_penalty == 'lin':
                for i in range(1,self.K_plus):
                    y   = self.adj_norm   @ y
                    y_t = self.adj_t_norm @ y

                    sum_src_to_dst = sum_src_to_dst + self.lins_src_to_dst[i](y)/i
                    sum_dst_to_src = sum_dst_to_src + self.lins_dst_to_src[i](y_t)/i
            else:
                raise ValueError(f"Weight penalty type {self.weight_penalty} not supported")
       
        total = self.alpha * sum_src_to_dst + (1 - self.alpha) * sum_dst_to_src


        return total


class ComplexFaberConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, alpha, K_plus=3,exponent = -0.25, weight_penalty = 'exp', zero_order = False):
        super(ComplexFaberConv, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.K_plus = K_plus
        self.exponent = exponent
        self.weight_penalty = weight_penalty
        self.zero_order = zero_order

        if zero_order:
            #Zero Order Lins
            #Source to destination 
            self.lin_real_src_to_dst_zero = Linear(input_dim, output_dim)
            self.lin_imag_src_to_dst_zero = Linear(input_dim, output_dim)

            #Destination to source
            self.lin_real_dst_to_src_zero = Linear(input_dim, output_dim)
            self.lin_imag_dst_to_src_zero = Linear(input_dim, output_dim)



        #Lins for positive powers:
        #Source to destination 
        self.lins_real_src_to_dst = torch.nn.ModuleList([
            Linear(input_dim, output_dim) for _ in range(K_plus)
        ])                                                          # real part

        self.lins_imag_src_to_dst = torch.nn.ModuleList([
            Linear(input_dim, output_dim) for _ in range(K_plus)
        ])                                                          # imaginary part

        #Destination to source
        self.lins_real_dst_to_src = torch.nn.ModuleList([
            Linear(input_dim, output_dim) for _ in range(K_plus)
        ])                                                          # real part
        self.lins_imag_dst_to_src = torch.nn.ModuleList([
            Linear(input_dim, output_dim) for _ in range(K_plus)
        ])                                                          # imaginary part
        self.alpha = alpha
        self.adj_norm, self.adj_t_norm = None, None

    def forward(self, x_real, x_imag, edge_index):
        if self.adj_norm is None:
            row, col = edge_index
            num_nodes = x_real.shape[0]

            adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
            self.adj_norm = get_norm_adj(adj, norm="dir", exponent = self.exponent)

            adj_t = SparseTensor(row=col, col=row, sparse_sizes=(num_nodes, num_nodes))
            self.adj_t_norm = get_norm_adj(adj_t, norm="dir", exponent = self.exponent)


        y_real = self.adj_norm   @ x_real
        y_imag = self.adj_norm   @ x_imag

        y_real_t = self.adj_t_norm @ x_real
        y_imag_t = self.adj_t_norm @ x_imag
        
        sum_real_src_to_dst = self.lins_real_src_to_dst[0](y_real) - self.lins_imag_src_to_dst[0](y_imag)  
        sum_imag_src_to_dst = self.lins_imag_src_to_dst[0](y_real) + self.lins_real_src_to_dst[0](y_imag)  
        sum_real_dst_to_src = self.lins_real_src_to_dst[0](y_real_t) - self.lins_imag_src_to_dst[0](y_imag_t) 
        sum_imag_dst_to_src = self.lins_imag_src_to_dst[0](y_real) + self.lins_real_src_to_dst[0](y_imag_t) 
        if self.zero_order:
            sum_real_src_to_dst = sum_real_src_to_dst + self.lin_real_src_to_dst_zero(x_real) - self.lin_imag_src_to_dst_zero(x_imag)
            sum_imag_src_to_dst = sum_imag_src_to_dst + self.lin_imag_src_to_dst_zero(x_real) + self.lin_real_src_to_dst_zero(x_imag)

            sum_real_dst_to_src = sum_real_dst_to_src + self.lin_real_dst_to_src_zero(x_real) - self.lin_imag_dst_to_src_zero(x_imag)
            sum_imag_dst_to_src = sum_imag_dst_to_src + self.lin_imag_dst_to_src_zero(x_real) + self.lin_real_dst_to_src_zero(x_imag)




        
        if self.K_plus > 1:
            if self.weight_penalty == 'exp':
                for i in range(1,self.K_plus):
                    y_real = self.adj_norm   @ x_real
                    y_imag = self.adj_norm   @ x_imag

                    y_real_t = self.adj_t_norm @ x_real
                    y_imag_t = self.adj_t_norm @ x_imag

                    sum_real_src_to_dst = sum_real_src_to_dst + (self.lins_real_src_to_dst[i](y_real) - self.lins_imag_src_to_dst[i](y_imag))/(2**i)
                    sum_imag_src_to_dst = sum_imag_src_to_dst + (self.lins_imag_src_to_dst[i](y_real) + self.lins_real_src_to_dst[i](y_imag))/(2**i)


                    sum_real_dst_to_src = sum_real_dst_to_src + (self.lins_real_src_to_dst[i](y_real_t) - self.lins_imag_src_to_dst[i](y_imag_t))/(2**i)
                    sum_imag_dst_to_src = sum_imag_dst_to_src + (self.lins_imag_src_to_dst[i](y_real) + self.lins_real_src_to_dst[i](y_imag_t))/(2**i)
            elif self.weight_penalty == 'lin':
                for i in range(1,self.K_plus):
                    y_real = self.adj_norm   @ x_real
                    y_imag = self.adj_norm   @ x_imag

                    y_real_t = self.adj_t_norm @ x_real
                    y_imag_t = self.adj_t_norm @ x_imag

                    sum_real_src_to_dst = sum_real_src_to_dst + (self.lins_real_src_to_dst[i](y_real) - self.lins_imag_src_to_dst[i](y_imag))/i
                    sum_imag_src_to_dst = sum_imag_src_to_dst + (self.lins_imag_src_to_dst[i](y_real) + self.lins_real_src_to_dst[i](y_imag))/i


                    sum_real_dst_to_src = sum_real_dst_to_src + (self.lins_real_src_to_dst[i](y_real_t) - self.lins_imag_src_to_dst[i](y_imag_t))/i
                    sum_imag_dst_to_src = sum_imag_dst_to_src + (self.lins_imag_src_to_dst[i](y_real) + self.lins_real_src_to_dst[i](y_imag_t))/i
            else:
                raise ValueError(f"Weight penalty type {self.weight_penalty} not supported")

       
        total_real = self.alpha * sum_real_src_to_dst + (1 - self.alpha) * sum_real_dst_to_src
        total_imag = self.alpha * sum_imag_src_to_dst + (1 - self.alpha) * sum_imag_dst_to_src


        return total_real, total_imag



class GNN(torch.nn.Module):
    def __init__(
        self,
        num_features,
        num_classes,
        hidden_dim,
        num_layers=2,
        dropout=0,
        conv_type="fabernet",
        jumping_knowledge=False,
        normalize=False,
        alpha=1 / 2,
        learn_alpha=False,
        K_plus = 3,
        exponent = -0.25,
        weight_penalty = 'exp',
        lrelu_slope = -1.0,
        zero_order = False,     
    ):
        super(GNN, self).__init__()
        self.conv_type = conv_type
        self.alpha = nn.Parameter(torch.ones(1) * alpha, requires_grad=learn_alpha)
        self.lrelu_slope = lrelu_slope
        

        output_dim = hidden_dim if jumping_knowledge else num_classes
        if num_layers == 1:
            self.convs = ModuleList([get_conv(conv_type, num_features, output_dim, alpha = self.alpha, K_plus = K_plus, zero_order = zero_order,exponent =  exponent,weight_penalty = weight_penalty)])
        else:
            self.convs = ModuleList([get_conv(conv_type, num_features, hidden_dim, alpha = self.alpha, K_plus = K_plus, zero_order = zero_order,exponent =  exponent,weight_penalty = weight_penalty)])
            for _ in range(num_layers - 2):
                self.convs.append(get_conv(conv_type, hidden_dim, hidden_dim, alpha = self.alpha, K_plus = K_plus, zero_order = zero_order,exponent =  exponent,weight_penalty = weight_penalty))
            self.convs.append(get_conv(conv_type, hidden_dim, output_dim,  alpha = self.alpha, K_plus = K_plus,  zero_order = zero_order,exponent =  exponent,weight_penalty = weight_penalty))

        if jumping_knowledge is not None:
            if self.conv_type == "complex-fabernet":
                input_dim = 2*hidden_dim * num_layers if jumping_knowledge == "cat" else 2*hidden_dim
            else:
                input_dim = hidden_dim * num_layers if jumping_knowledge == "cat" else hidden_dim
            self.lin = Linear(input_dim, num_classes)
            self.jump = JumpingKnowledge(mode=jumping_knowledge, channels=hidden_dim, num_layers=num_layers)

        self.num_layers = num_layers
        self.dropout = dropout
        self.jumping_knowledge = jumping_knowledge
        self.normalize = normalize


    

    def forward(self, x, edge_index):
        if self.conv_type == "complex-fabernet":
            x_real =  x

            x_imag = torch.zeros_like(x)

            xs = []
            for i, conv in enumerate(self.convs):
                x_real, x_imag = conv(x_real, x_imag, edge_index)
                if i != len(self.convs) - 1 or self.jumping_knowledge:
                    x_real = F.leaky_relu(x_real,negative_slope= self.lrelu_slope)
                    x_imag = F.leaky_relu(x_imag,negative_slope= self.lrelu_slope)

                    x_real = F.dropout(x_real, p=self.dropout, training=self.training)
                    x_imag = F.dropout(x_imag, p=self.dropout, training=self.training)
                    if self.normalize: 
                        x_real = F.normalize(x_real, p=2, dim=1)
                        x_imag = F.normalize(x_imag, p=2, dim=1)

                xs += [torch.cat((x_real,x_imag),1)]
            x = torch.cat((x_real,x_imag),1)

            if self.jumping_knowledge is not None:
                x = self.jump(xs)
                x = self.lin(x)

            return torch.nn.functional.log_softmax(x, dim=1)
        else:
            xs = []
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index)
                if i != len(self.convs) - 1 or self.jumping_knowledge:
                    x = F.leaky_relu(x,negative_slope= self.lrelu_slope)
                    x = F.dropout(x, p=self.dropout, training=self.training)
                    if self.normalize:
                        x = F.normalize(x, p=2, dim=1)
                xs += [x]

            if self.jumping_knowledge is not None:
                x = self.jump(xs)
                x = self.lin(x)

            return torch.nn.functional.log_softmax(x, dim=1)















class LightingFullBatchModelWrapper(pl.LightningModule):
    def __init__(self, model, lr, weight_decay, real_weight_decay, imag_weight_decay, train_mask, val_mask, test_mask, evaluator=None):
        super().__init__()
        self.model = model
        self.lr = lr
        self.imag_weight_decay = imag_weight_decay
        self.real_weight_decay = real_weight_decay
        self.weight_decay = weight_decay
        self.evaluator = evaluator
        self.train_mask, self.val_mask, self.test_mask = train_mask, val_mask, test_mask

    def training_step(self, batch, batch_idx):
        x, y, edge_index = batch.x, batch.y.long(), batch.edge_index
        out = self.model(x, edge_index)

        loss = nn.functional.nll_loss(out[self.train_mask], y[self.train_mask].squeeze())
        self.log("train_loss", loss)

        y_pred = out.max(1)[1]
        train_acc = self.evaluate(y_pred=y_pred[self.train_mask], y_true=y[self.train_mask])
        self.log("train_acc", train_acc)
        val_acc = self.evaluate(y_pred=y_pred[self.val_mask], y_true=y[self.val_mask])
        self.log("val_acc", val_acc)

        return loss

    def evaluate(self, y_pred, y_true):
        if self.evaluator:
            acc = self.evaluator.eval({"y_true": y_true, "y_pred": y_pred.unsqueeze(1)})["acc"]
        else:
            acc = y_pred.eq(y_true.squeeze()).sum().item() / y_pred.shape[0]

        return acc

    def test_step(self, batch, batch_idx):
        x, y, edge_index = batch.x, batch.y.long(), batch.edge_index
        out = self.model(x, edge_index)

        y_pred = out.max(1)[1]
        val_acc = self.evaluate(y_pred=y_pred[self.test_mask], y_true=y[self.test_mask])
        self.log("test_acc", val_acc)

    def configure_optimizers(self):

        imag_weights = list()
        real_weights = list()
        other_params = list()

        for name, param in self.model.named_parameters():
            if "imag" in name:
                imag_weights.append(param)
  
            elif "real" in name:
                real_weights.append(param)

            else:
                other_params.append(param)

        optimizer = optim.AdamW([{'params': other_params, 'weight_decay': self.weight_decay}, {'params': real_weights, 'weight_decay': self.real_weight_decay}, {'params': imag_weights, 'weight_decay': self.imag_weight_decay}], lr = self.lr)
        
        return optimizer


def get_model(args):
    return GNN(
        num_features=args.num_features,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=args.num_classes,
        dropout=args.dropout,
        conv_type=args.conv_type,
        jumping_knowledge=args.jk,
        normalize=args.normalize,
        alpha=args.alpha,
        learn_alpha=args.learn_alpha,
        K_plus = args.k_plus,
        zero_order = args.zero_order,
        exponent = args.exponent,
        weight_penalty = args.weight_penalty,
        lrelu_slope= args.lrelu_slope,
    )

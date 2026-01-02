import torch
from torch import nn, optim
import pytorch_lightning as pl
from torch_sparse import SparseTensor
import torch.nn.functional as F
from torch.nn import ModuleList, Linear
from torch_geometric.nn import JumpingKnowledge
from torch_geometric.utils import degree
from torch_scatter import scatter_add

from torchmetrics.classification import (
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)

from src.datasets.data_utils import get_norm_adj


# ---------------------------------------------------------------------------
# Convolution selector
# ---------------------------------------------------------------------------


def get_conv(conv_type, input_dim, output_dim, alpha):
    if conv_type == "dir-gcn":
        return DirGCNConv(input_dim, output_dim, alpha)
    elif conv_type == "dir-gcn-gated":
        return GatedDirGCNConv(input_dim, output_dim, alpha)
    else:
        raise ValueError(f"Convolution type {conv_type} not supported")


# ---------------------------------------------------------------------------
# Baseline Dir-GCN convolution
# ---------------------------------------------------------------------------


class DirGCNConv(torch.nn.Module):
    """
    Baseline Dir-GCN layer.

    Uses a global (possibly learnable) alpha to mix:
      - src→dst (forward) messages
      - dst→src (reverse) messages
    """

    def __init__(self, input_dim, output_dim, alpha):
        super(DirGCNConv, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.lin_src_to_dst = Linear(input_dim, output_dim)
        self.lin_dst_to_src = Linear(input_dim, output_dim)
        self.alpha = alpha
        self.adj_norm, self.adj_t_norm = None, None

    def forward(self, x, edge_index):
        if self.adj_norm is None:
            row, col = edge_index
            num_nodes = x.shape[0]

            adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
            self.adj_norm = get_norm_adj(adj, norm="dir")

            adj_t = SparseTensor(row=col, col=row, sparse_sizes=(num_nodes, num_nodes))
            self.adj_t_norm = get_norm_adj(adj_t, norm="dir")

        return self.alpha * self.lin_src_to_dst(self.adj_norm @ x) + (
            1 - self.alpha
        ) * self.lin_dst_to_src(self.adj_t_norm @ x)


# ---------------------------------------------------------------------------
# Enhanced Dir-GCN with edge-wise contribution scores + gating + residual
# ---------------------------------------------------------------------------


class GatedDirGCNConv(torch.nn.Module):
    """
    Enhanced Dir-GCN layer.

    Differences vs baseline:
      - Computes learnable edge-wise contribution scores a_ij
      - Aggregates messages along src→dst and dst→src using these scores
      - Normalizes by (in/out) degree for stability
      - Learns a node-wise gate g ∈ (0,1) to fuse the two directional
        aggregations
      - Adds a residual connection from the original node features
    """

    def __init__(self, input_dim, output_dim, alpha):
        super(GatedDirGCNConv, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Direction-specific projections (like baseline)
        self.lin_src_to_dst = Linear(input_dim, output_dim)
        self.lin_dst_to_src = Linear(input_dim, output_dim)

        # Edge scoring network: uses original node features (x_src, x_dst)
        # to compute a local contribution score for each edge.
        # a_ij = sigmoid(edge_mlp([x_i || x_j]))
        self.edge_mlp = nn.Sequential(
            Linear(2 * input_dim, input_dim),
            nn.ReLU(),
            Linear(input_dim, 1),
        )

        # Node-wise gate: [h_in || h_out] -> g ∈ (0,1)
        self.gate_mlp = nn.Sequential(
            Linear(2 * output_dim, output_dim),
            nn.ReLU(),
            Linear(output_dim, 1),
        )

        # Residual connection on x
        self.residual = (
            Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        )

        # alpha kept only for compatibility with get_conv API
        self.alpha = alpha

    def forward(self, x, edge_index):
        """
        x: [N, F]
        edge_index: [2, E] (directed edges, src->dst)
        """
        num_nodes = x.size(0)
        src, dst = edge_index  # [E], [E]

        # --------------------------------------------------
        # 1) Edge-wise contribution scores (local importance)
        # --------------------------------------------------
        x_src = x[src]  # [E, F]
        x_dst = x[dst]  # [E, F]

        edge_input = torch.cat([x_src, x_dst], dim=-1)  # [E, 2F]
        edge_score = torch.sigmoid(self.edge_mlp(edge_input)).squeeze(-1)  # [E]

        # --------------------------------------------------
        # 2) Directional messages
        # --------------------------------------------------
        # src→dst messages: project source features
        msg_in = self.lin_src_to_dst(x_src)  # [E, D]

        # dst→src messages: project destination features
        msg_out = self.lin_dst_to_src(x_dst)  # [E, D]

        # weight messages by local contribution scores
        msg_in = msg_in * edge_score.unsqueeze(-1)  # [E, D]
        msg_out = msg_out * edge_score.unsqueeze(-1)  # [E, D]

        # --------------------------------------------------
        # 3) Aggregate with degree normalization
        # --------------------------------------------------
        # in-direction: aggregate messages into dst
        h_in = scatter_add(msg_in, dst, dim=0, dim_size=num_nodes)  # [N, D]

        # out-direction (reverse): aggregate messages into src
        h_out = scatter_add(msg_out, src, dim=0, dim_size=num_nodes)  # [N, D]

        # Normalize by degrees for scale stability
        in_deg = degree(dst, num_nodes=num_nodes, dtype=torch.float32).clamp(min=1.0)
        out_deg = degree(src, num_nodes=num_nodes, dtype=torch.float32).clamp(min=1.0)

        h_in = h_in / in_deg.unsqueeze(-1)
        h_out = h_out / out_deg.unsqueeze(-1)

        # --------------------------------------------------
        # 4) Node-wise gate between directional aggregations
        # --------------------------------------------------
        gate_input = torch.cat([h_in, h_out], dim=-1)  # [N, 2D]
        g = torch.sigmoid(self.gate_mlp(gate_input))  # [N, 1]

        fused = g * h_in + (1.0 - g) * h_out  # [N, D]

        # --------------------------------------------------
        # 5) Residual connection
        # --------------------------------------------------
        res = self.residual(x)  # [N, D]
        out = fused + res

        return out


# ---------------------------------------------------------------------------
# Multi-layer GNN wrapper
# ---------------------------------------------------------------------------


class GNN(torch.nn.Module):
    def __init__(
        self,
        num_features,
        num_classes,
        hidden_dim,
        num_layers=2,
        dropout=0,
        conv_type="dir-gcn",
        jumping_knowledge=False,
        normalize=False,
        alpha=1 / 2,
        learn_alpha=False,
    ):
        super(GNN, self).__init__()

        self.num_classes = num_classes  # for metrics
        self.alpha = nn.Parameter(torch.ones(1) * alpha, requires_grad=learn_alpha)

        output_dim = hidden_dim if jumping_knowledge else num_classes
        if num_layers == 1:
            self.convs = ModuleList(
                [get_conv(conv_type, num_features, output_dim, self.alpha)]
            )
        else:
            self.convs = ModuleList(
                [get_conv(conv_type, num_features, hidden_dim, self.alpha)]
            )
            for _ in range(num_layers - 2):
                self.convs.append(
                    get_conv(conv_type, hidden_dim, hidden_dim, self.alpha)
                )
            self.convs.append(get_conv(conv_type, hidden_dim, output_dim, self.alpha))

        if jumping_knowledge is not None:
            input_dim = (
                hidden_dim * num_layers if jumping_knowledge == "cat" else hidden_dim
            )
            self.lin = Linear(input_dim, num_classes)
            self.jump = JumpingKnowledge(
                mode=jumping_knowledge, channels=hidden_dim, num_layers=num_layers
            )

        self.num_layers = num_layers
        self.dropout = dropout
        self.jumping_knowledge = jumping_knowledge
        self.normalize = normalize

    def forward(self, x, edge_index):
        xs = []
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1 or self.jumping_knowledge:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                if self.normalize:
                    x = F.normalize(x, p=2, dim=1)
            xs += [x]

        if self.jumping_knowledge is not None:
            x = self.jump(xs)
            x = self.lin(x)

        return torch.nn.functional.log_softmax(x, dim=1)


# ---------------------------------------------------------------------------
# Lightning wrapper with metrics
# ---------------------------------------------------------------------------


class LightingFullBatchModelWrapper(pl.LightningModule):
    def __init__(
        self, model, lr, weight_decay, train_mask, val_mask, test_mask, evaluator=None
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.evaluator = evaluator
        self.train_mask, self.val_mask, self.test_mask = (
            train_mask,
            val_mask,
            test_mask,
        )

        # ====== Metrics (macro over classes) ======
        num_classes = int(getattr(self.model, "num_classes", 2))

        # train metrics
        self.train_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.train_prec = MulticlassPrecision(num_classes=num_classes, average="macro")
        self.train_rec = MulticlassRecall(num_classes=num_classes, average="macro")

        # val metrics
        self.val_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.val_prec = MulticlassPrecision(num_classes=num_classes, average="macro")
        self.val_rec = MulticlassRecall(num_classes=num_classes, average="macro")

        # test metrics
        self.test_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.test_prec = MulticlassPrecision(num_classes=num_classes, average="macro")
        self.test_rec = MulticlassRecall(num_classes=num_classes, average="macro")

    def training_step(self, batch, batch_idx):
        x, y, edge_index = batch.x, batch.y.long(), batch.edge_index
        out = self.model(x, edge_index)

        loss = nn.functional.nll_loss(
            out[self.train_mask], y[self.train_mask].squeeze()
        )
        self.log("train_loss", loss)

        y_pred = out.max(1)[1]

        # ----- train metrics -----
        self.log(
            "train_f1",
            self.train_f1(y_pred[self.train_mask], y[self.train_mask].squeeze()),
            prog_bar=True,
        )
        self.log(
            "train_prec",
            self.train_prec(y_pred[self.train_mask], y[self.train_mask].squeeze()),
        )
        self.log(
            "train_rec",
            self.train_rec(y_pred[self.train_mask], y[self.train_mask].squeeze()),
        )

        train_acc = self.evaluate(
            y_pred=y_pred[self.train_mask], y_true=y[self.train_mask]
        )
        self.log("train_acc", train_acc, prog_bar=True)

        # ----- val metrics -----
        self.log(
            "val_f1",
            self.val_f1(y_pred[self.val_mask], y[self.val_mask].squeeze()),
            prog_bar=True,
        )
        self.log(
            "val_prec",
            self.val_prec(y_pred[self.val_mask], y[self.val_mask].squeeze()),
        )
        self.log(
            "val_rec",
            self.val_rec(y_pred[self.val_mask], y[self.val_mask].squeeze()),
        )

        val_acc = self.evaluate(y_pred=y_pred[self.val_mask], y_true=y[self.val_mask])
        self.log("val_acc", val_acc, prog_bar=True)

        return loss

    def evaluate(self, y_pred, y_true):
        if self.evaluator:
            acc = self.evaluator.eval(
                {"y_true": y_true, "y_pred": y_pred.unsqueeze(1)}
            )["acc"]
        else:
            acc = y_pred.eq(y_true.squeeze()).sum().item() / y_pred.shape[0]
        return acc

    def test_step(self, batch, batch_idx):
        x, y, edge_index = batch.x, batch.y.long(), batch.edge_index
        out = self.model(x, edge_index)

        y_pred = out.max(1)[1]

        # ----- test metrics -----
        self.log(
            "test_f1",
            self.test_f1(y_pred[self.test_mask], y[self.test_mask].squeeze()),
        )
        self.log(
            "test_prec",
            self.test_prec(y_pred[self.test_mask], y[self.test_mask].squeeze()),
        )
        self.log(
            "test_rec",
            self.test_rec(y_pred[self.test_mask], y[self.test_mask].squeeze()),
        )

        val_acc = self.evaluate(y_pred=y_pred[self.test_mask], y_true=y[self.test_mask])
        self.log("test_acc", val_acc)

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_model(args):
    return GNN(
        num_features=args.num_features,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=args.num_classes,
        dropout=args.dropout,
        conv_type=args.conv_type,  # "dir-gcn" or "dir-gcn-gated"
        jumping_knowledge=args.jk,
        normalize=args.normalize,
        alpha=args.alpha,
        learn_alpha=args.learn_alpha,
    )

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from backbones.bert_model.TextEncoder import TextEncoder_Bert
from nets.causal_reasoning import SCMReasoningHead
from nets.unet_blocks import ConvBatchNorm, DownBlock, NLBlock, UpBlock


SEGFORMER_B0_MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"


class SegFormerB0Bottleneck(nn.Module):
    """Project SegFormer-B0's final encoder stage into EviVLM's bottleneck shape."""

    def __init__(
        self,
        model_name=SEGFORMER_B0_MODEL_NAME,
        out_channels=512,
        out_size=(14, 14),
        pretrained=True,
        freeze_encoder=False,
    ):
        super().__init__()
        try:
            from transformers import SegformerConfig, SegformerModel
        except ImportError as exc:
            raise ImportError(
                "SegFormer support requires transformers. Install dependencies with "
                "`pip install -r requirements.txt`."
            ) from exc

        if pretrained:
            self.encoder = SegformerModel.from_pretrained(
                model_name,
                output_hidden_states=True,
                ignore_mismatched_sizes=True,
            )
        else:
            self.encoder = SegformerModel(
                SegformerConfig(output_hidden_states=True)
            )

        hidden_size = self.encoder.config.hidden_sizes[-1]
        self.out_size = out_size
        self.freeze_encoder = freeze_encoder
        self.project = nn.Sequential(
            nn.Conv2d(hidden_size, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def forward(self, x):
        x = (x - self.image_mean.type_as(x)) / self.image_std.type_as(x)
        if self.freeze_encoder:
            with torch.no_grad():
                outputs = self.encoder(
                    pixel_values=x,
                    output_hidden_states=True,
                    return_dict=True,
                )
        else:
            outputs = self.encoder(
                pixel_values=x,
                output_hidden_states=True,
                return_dict=True,
            )
        features = outputs.hidden_states[-1]
        features = self.project(features)
        if features.shape[-2:] != self.out_size:
            features = F.interpolate(
                features,
                size=self.out_size,
                mode="bilinear",
                align_corners=False,
            )
        return features


def exp_evidence(y, temp=0.8):
    return torch.exp(torch.div(torch.clamp(y, -10, 10), temp))


def get_p_and_u_from_logit(x):
    alpha = exp_evidence(x) + torch.ones_like(x)  # [2, 25, 2]
    p = alpha[..., 0] / torch.sum(alpha, dim=-1)
    u = 2 / torch.sum(alpha, dim=-1)
    return p, u


class EviVLM(nn.Module):
    def __init__(
        self,
        n_channels=3,
        n_classes=1,
        vision_backbone="unet",
        segformer_model_name=SEGFORMER_B0_MODEL_NAME,
        segformer_pretrained=True,
        freeze_segformer=False,
    ):
        super().__init__()
        self.device = torch.device("cuda:0")
        self.temperature = 0.07
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.vision_backbone = vision_backbone.lower()
        in_channels = 64
        self.inc = ConvBatchNorm(n_channels, in_channels)
        self.down1 = DownBlock(in_channels, in_channels * 2, nb_Conv=2)
        self.down2 = DownBlock(in_channels * 2, in_channels * 4, nb_Conv=2)
        self.down3 = DownBlock(in_channels * 4, in_channels * 8, nb_Conv=2)
        self.down4 = DownBlock(in_channels * 8, in_channels * 8, nb_Conv=2)
        if self.vision_backbone in {"segformer", "segformer-b0"}:
            self.segformer_bottleneck = SegFormerB0Bottleneck(
                model_name=segformer_model_name,
                out_channels=in_channels * 8,
                out_size=(14, 14),
                pretrained=segformer_pretrained,
                freeze_encoder=freeze_segformer,
            )
        elif self.vision_backbone != "unet":
            raise ValueError(
                "vision_backbone must be one of: 'unet', 'segformer', 'segformer-b0'"
            )
        self.up4 = UpBlock(in_channels * 16, in_channels * 4, nb_Conv=2)
        self.up3 = UpBlock(in_channels * 8, in_channels * 2, nb_Conv=2)
        self.up2 = UpBlock(in_channels * 4, in_channels, nb_Conv=2)
        self.up1 = UpBlock(in_channels * 2, in_channels, nb_Conv=2)
        if n_classes == 1:
            self.last_activation = nn.Sigmoid()
        else:
            self.last_activation = None

        self.text_encoder = TextEncoder_Bert()
        self.causal_reasoner = SCMReasoningHead()

        self.V2L = nn.Sequential(
            # nn.Linear(64, 64),
            # nn.Dropout(0.2),
            nn.LeakyReLU(0.2),
        )

        self.L2V = nn.Sequential(
            # nn.Linear(64, 64),
            # nn.Dropout(0.2),
            nn.LeakyReLU(0.2),
        )

        self.prob_alpha = nn.Conv2d(in_channels, n_classes, kernel_size=(1, 1))
        self.prob_beta = nn.Conv2d(in_channels, n_classes, kernel_size=(1, 1))

        self.prob_alpha_2 = nn.Conv2d(in_channels, 2, kernel_size=(1, 1))
        self.prob_beta_2 = nn.Conv2d(in_channels, 2, kernel_size=(1, 1))

        self.vision_nonlocal = NLBlock(in_channels=512, inter_channels=512)
        self.text_nonlocal = NLBlock(in_channels=512, inter_channels=512)

        self.cross_att = nn.Sequential(
            nn.Conv2d(2, 2, (3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(2, 2, (1, 1)),
            nn.Softmax(dim=1),
        )

    def forward(self, x, texts, return_reasoning=False):
        b = x.shape[0]
        x = x.float()
        self.device = x.device
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)  # [b, 512, 14, 14]
        if self.vision_backbone in {"segformer", "segformer-b0"}:
            x5 = self.segformer_bottleneck(x)  # [b, 512, 14, 14]

        report_feat, word_feat, _, sents = self.text_encoder(texts, self.device)
        report_emb = self.text_encoder.global_embed(report_feat)
        report_emb = F.normalize(report_emb, dim=-1)  # [b, 512]
        word_emb = self.text_encoder.local_embed(word_feat)  # [b, 19, 512]
        word_emb = F.normalize(word_emb, dim=-1)
        x5_nonlocal = self.vision_nonlocal(x5)  # [b, 196, 196]

        patch_emb = x5.view(b, 512, -1).permute(0, 2, 1)  # [b, 196, 512]
        img_emb_V = torch.mean(patch_emb, dim=1)  # [b, 512]
        img_emb_V = F.normalize(img_emb_V, dim=-1)  # [b, 512]

        patch_emb = F.normalize(patch_emb, dim=-1)
        mask = (
            torch.from_numpy(np.array(sents)[:, 1:] == "[PAD]")
            .type_as(patch_emb)
            .bool()
        )
        atten_sim = torch.bmm(patch_emb, word_emb.permute(0, 2, 1))  # [b, 196, 9]
        patch_num = patch_emb.size(1)  # 196
        atten_sim[mask.unsqueeze(1).repeat(1, patch_num, 1)] = float(
            "-inf"
        )  # [b, 196, 9]
        atten_scores = F.softmax(atten_sim / self.temperature, dim=-1)  # [b, 196, 9]
        patch_emb_atten = torch.bmm(atten_scores, word_emb)  # [b, 196, 512]
        x5_2 = patch_emb_atten.permute(0, 2, 1).view(b, 512, 14, 14)  # [b, 512, 14, 14]
        x5_2_nonlocal = self.text_nonlocal(x5_2)  # [b, 196, 196]

        # affinity matrix
        if len(x5_nonlocal.size()) != 4:
            x5_nonlocal = x5_nonlocal.unsqueeze(1)
            x5_2_nonlocal = x5_2_nonlocal.unsqueeze(1)
        cross_aff = torch.cat((x5_nonlocal, x5_2_nonlocal), dim=1)
        cross_w = self.cross_att(cross_aff)
        cross_aff = cross_aff[:, 0] * cross_w[:, 0] + cross_aff[:, 1] * cross_w[:, 1]

        refined_x5 = torch.matmul(cross_aff, patch_emb)  # [b, 196, 512]
        refined_x5_2 = torch.matmul(cross_aff, patch_emb_atten)  # [b, 196, 512]
        refined_x5_affinity = refined_x5.permute(0, 2, 1).view(
            b, 512, 14, 14
        )  # [b, 512, 14, 14]
        refined_x5_2_affinity = refined_x5_2.permute(0, 2, 1).view(
            b, 512, 14, 14
        )  # [b, 512, 14, 14]
        x5 = x5 * refined_x5_2_affinity
        x5_2 = x5_2 * refined_x5_2_affinity

        img_emb_L = torch.mean(patch_emb_atten, dim=1)  # [b, 512]
        img_emb_L = F.normalize(img_emb_L, dim=-1)  # [b, 512]

        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x_V = self.up1(x, x1)  # [b, 64, 224, 224]

        x_2 = self.up4(x5_2, x4)
        x_2 = self.up3(x_2, x3)
        x_2 = self.up2(x_2, x2)
        x_L = self.up1(x_2, x1)  # [b, 64, 224, 224]

        alpha_V = self.prob_alpha(x_V)  # [b, 64, 224, 224]-->[b, 1, 224, 224]
        ##################################################################################################################
        # alpha_V_64 = x_V.permute(0, 2, 3, 1).reshape(b * 224 * 224, 64)  # [n, 64]
        # alpha_V_64 = F.normalize(alpha_V_64, dim=-1)

        alpha_V_2 = self.prob_alpha_2(x_V)  # [b, 64, 224, 224]-->[b, 2, 224, 224]
        alpha_V_2 = alpha_V_2.permute(0, 2, 3, 1).reshape(b * 224 * 224, 2)
        alpha_V_2 = nn.Softplus()(alpha_V_2)  # [n, 2]
        ##################################################################################################################
        # alpha_L = self.prob_alpha(x_L)  # [b, 64, 224, 224]-->[b, 1, 224, 224]

        # x_V2L = x_V + self.V2L((x_V + x_L).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # x_L2V = x_L + self.L2V((x_V + x_L).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # beta_V = self.prob_beta(x_V2L)  # [b, 64, 224, 224]-->[b, 1, 224, 224]
        beta_L = self.prob_beta(x_L)  # [b, 64, 224, 224]-->[b, 1, 224, 224]
        ##################################################################################################################
        # beta_L_64 = x_L.permute(0, 2, 3, 1).reshape(b*224*224, 64)  # [n, 64]
        # beta_L_64 = F.normalize(beta_L_64, dim=-1)

        beta_L_2 = self.prob_alpha_2(x_L)  # [b, 64, 224, 224]-->[b, 2, 224, 224]
        beta_L_2 = beta_L_2.permute(0, 2, 3, 1).reshape(b * 224 * 224, 2)
        beta_L_2 = nn.Softplus()(beta_L_2)  # [n, 2]
        ##################################################################################################################

        # logit_V = torch.stack((alpha_V, beta_V), dim=-1)  # [b, 1, 224, 224, 2]
        # logit_L = torch.stack((alpha_L, beta_L), dim=-1)  # [b, 1, 224, 224, 2]
        #
        # prob_V, uct_V = get_p_and_u_from_logit(logit_V)  # [b, 1, 224, 224, 2]-->[b, 1, 224, 224]
        # prob_L, uct_L = get_p_and_u_from_logit(logit_L)  # [b, 1, 224, 224, 2]-->[b, 1, 224, 224]

        prob_VL = (alpha_V + beta_L) / 2.0
        # uct_VL = (uct_V + uct_L) / 2.0
        prob_V = self.last_activation(alpha_V)
        prob_L = self.last_activation(beta_L)
        prob_VL = self.last_activation(prob_VL)

        evi_V = alpha_V_2
        evi_L = beta_L_2
        evi_VL = (alpha_V_2 + beta_L_2) / 2.0

        ##################################################################################################################
        S_V = torch.sum(alpha_V_2, dim=1)
        un_V = 2 / S_V
        un_V = un_V.view(-1, 224 * 224)
        un_V = torch.mean(un_V, dim=1)
        un_V = torch.sigmoid(un_V)

        S_L = torch.sum(beta_L_2, dim=1)
        un_L = 2 / S_L
        un_L = un_L.view(-1, 224 * 224)
        un_L = torch.mean(un_L, dim=1)
        un_L = torch.sigmoid(un_L)
        # -----------------------------------------------------------------------------------------------------------------
        affinity_V = un_V * img_emb_V.mm(un_L * img_emb_L.t())
        affinity_L = un_L * img_emb_L.mm(un_V * img_emb_V.t())
        labels = torch.arange(b).type_as(report_emb).long()
        scores = un_V * img_emb_V.mm(un_L * img_emb_L.t())
        scores = scores / self.temperature
        scores = torch.nan_to_num(scores, nan=0.0, posinf=1e4, neginf=-1e4)
        scores1 = scores.transpose(0, 1)
        loss0 = F.cross_entropy(scores, labels)
        loss1 = F.cross_entropy(scores1, labels)
        loss_0_1 = (loss0 + loss1) / 2.0

        E_s = torch.mean(affinity_V)
        E_t = torch.mean(affinity_L)
        Var_s = torch.var(affinity_V, unbiased=False)
        Var_t = torch.var(affinity_L, unbiased=False)

        s_flat = affinity_V.flatten()
        t_flat = affinity_L.flatten()
        Cov_st = torch.mean((s_flat - E_s) * (t_flat - E_t))
        # Bias-Variance Decomposition
        N = 32
        Bias = b**2 * (E_s - E_t) ** 2
        Variance = b**2 * (Var_s + Var_t - 2 * Cov_st)
        diff_loss = Bias + Variance

        loss_sim = loss_0_1 * 1.0 + diff_loss * 0.1
        loss_sim = torch.nan_to_num(loss_sim, nan=0.0, posinf=1e4, neginf=1e4)

        ##################################################################################################################

        # y = alpha_V
        # y = self.last_activation(y)

        if return_reasoning:
            reasoning_outputs = self.causal_reasoner(
                word_emb=word_emb,
                token_words=[sentence[1:] for sentence in sents],
                patch_emb=patch_emb,
                atten_scores=atten_scores,
                seg_prob=prob_VL,
                evidence=evi_VL,
            )
            return (
                prob_V,
                prob_L,
                prob_VL,
                evi_V,
                evi_L,
                evi_VL,
                loss_sim,
                reasoning_outputs,
            )

        return prob_V, prob_L, prob_VL, evi_V, evi_L, evi_VL, loss_sim


# device = torch.device('cuda:0')
# model = UNet().to(device)
# img = torch.rand(4, 3, 224, 224).to(device)
# text = ['Bilateral pulmonary infection, two infected areas, middle lower left lung and upper middle lower right lung.', 'my dog', 'my paper', 'vision language model']
# model(img, text)

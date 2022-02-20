import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import ArcMarginProduct, ArcMarginProduct2
from timm.models.layers import get_act_layer

from triplet_attention import TripletAttention


class GeMP(nn.Module):
    def __init__(self, p=3., eps=1e-6, learn_p=False):
        super().__init__()
        self._p = p
        self._learn_p = learn_p
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        self.set_learn_p(flag=learn_p)

    def set_learn_p(self, flag):
        self._learn_p = flag
        self.p.requires_grad = flag

    def forward(self, x):
        x = F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            (x.size(-2), x.size(-1))
        ).pow(1.0 / self.p)

        return x


class BlockAttentionModel(nn.Module):
    def __init__(
        self,
        num_calss_id,
        num_calss,
        backbone: nn.Module,
        n_features: int,
        margin=0.0,
        s=30.0,
    ):
        """Initialize"""
        super(BlockAttentionModel, self).__init__()
        self.backbone = backbone
        self.n_features = n_features
        self.drop_rate = 0.2
        self.pooling = 'gem'
        act_layer = nn.ReLU
        

        self.attention = TripletAttention(self.n_features,
                                              act_layer=act_layer,
                                              kernel_size=13)

        if self.pooling == 'avg':
            self.global_pool = torch.nn.AdaptiveAvgPool2d(1)
        elif self.pooling == 'gem':
            self.global_pool = GeMP(p=4.0, learn_p=False)
        elif self.pooling == 'max':
            self.global_pool = torch.nn.AdaptiveMaxPool2d(1)
        elif self.pooling == 'nop':
            self.global_pool = torch.nn.Identity()
        else:
            raise NotImplementedError(f'Invalid pooling type: {self.pooling}')
        fc_dim = 512
        self.fc = nn.Linear(self.n_features, fc_dim)
        self.bn = nn.BatchNorm1d(fc_dim)    
        self.face_margin_product = ArcMarginProduct(fc_dim, num_calss_id, s=s, m=margin)
        self.head = nn.Linear(fc_dim, num_calss_id)
        self.head2 = nn.Linear(fc_dim, num_calss)

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        if type(self.fc.bias) == torch.nn.parameter.Parameter:
            nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)
        
    def extract_feature(self, x):
        x = self.backbone(x)
        x = self.attention(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn(x)
        return x


    def forward(self, x , label):
        """Forward"""
        feature  = self.extract_feature(x)
        out_face = self.face_margin_product(feature, label)
        y = self.head(feature)
        z = self.head2(feature)
        return y,z,out_face


def get_model(backbone_name='tf_efficientnet_b0_ns',num_calss_id=15587,num_calss=30):
    # create backbone

    if backbone_name in ['tf_efficientnet_b0_ns', 'tf_efficientnet_b1_ns',
                         'tf_efficientnet_b2_ns', 'tf_efficientnet_b3_ns',
                         'tf_efficientnet_b4_ns', 'tf_efficientnet_b5_ns',
                         'tf_efficientnet_b6_ns', 'tf_efficientnet_b7_ns']:
        kwargs = {}
        if True:
            act_layer = get_act_layer('swish')
            kwargs['act_layer'] = act_layer

        backbone = timm.create_model(backbone_name, pretrained=True,in_chans=3, **kwargs)
        n_features = backbone.num_features
        backbone.reset_classifier(0, '')

    else:
        raise NotImplementedError(f'not implemented yet: {backbone_name}')

    model = BlockAttentionModel(backbone = backbone, n_features = n_features , num_calss_id=num_calss_id,num_calss=num_calss)

    return model

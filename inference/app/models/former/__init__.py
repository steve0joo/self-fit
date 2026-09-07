"""Former-DFER (Zhao & Liu, ACM MM 2021). AI-Hub NIA 베이스라인의 ST_Former.py 를 패키지로 옮김.
s_former.py / t_former.py 는 원본 그대로이며 import 경로만 상대 경로로 바꿨다."""

from torch import nn

from .s_former import spatial_transformer
from .t_former import temporal_transformer


class GenerateModel(nn.Module):
    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.s_former = spatial_transformer()
        self.t_former = temporal_transformer()
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):  # x: (B, 16, 3, 112, 112)
        x = self.s_former(x)
        x = self.t_former(x)
        return self.fc(x)

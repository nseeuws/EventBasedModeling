from typing import Optional, NamedTuple

import torch
from torch import nn, Tensor 


class EventOutput(NamedTuple): 
    center: Tensor
    duration: Tensor
    center_logit: Tensor


class Event(nn.Module):
    def __init__(
        self, channels: int = 64, dropout: Optional[float] = None
    ) -> None:
        super().__init__()

        stride = 4
        prior = torch.tensor([1e-1,], dtype=torch.float32)
        bias = nn.parameter.Parameter(
            data=-torch.log((1 - prior) / prior)
        )


        # Utilities
        self.up = nn.Upsample(scale_factor=stride)
        self.sigm = nn.Sigmoid()

        # First patch
        # Stage 0
        self.enc0 = nn.Sequential(
            nn.Conv1d(
                in_channels=1, out_channels=channels // 2,
                stride=1, kernel_size=20, padding='same'
            ),
            nn.ELU(inplace=True)
        )
        self.down = nn.MaxPool1d(kernel_size=stride)

        # Stage 1
        self.enc1 = nn.Sequential(
            nn.Conv1d(
                in_channels=channels // 2, out_channels=channels,
                kernel_size=20, stride=1, bias=False, padding='same'
            ),
            nn.BatchNorm1d(num_features=channels),
            nn.ELU(inplace=True)
        )

        # Stage 2
        self.enc2 = nn.Sequential(
            nn.Conv1d(
                in_channels=channels, out_channels=channels,
                kernel_size=15, stride=1, bias=False, padding='same'
            ),
            nn.BatchNorm1d(num_features=channels),
            nn.ELU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.Conv1d(
                in_channels=2 * channels, out_channels=channels,
                kernel_size=15, stride=1, bias=False, padding='same'
            ),
            nn.BatchNorm1d(num_features=channels),
            nn.ELU(),
            nn.Conv1d(
                in_channels=channels, out_channels=channels,
                kernel_size=15, stride=1, bias=False, padding='same'
            ),
            nn.BatchNorm1d(num_features=channels),
            nn.ELU(),
        )

        # Stage 3
        self.enc3 = nn.Sequential(
            nn.Conv1d(
                in_channels=channels, out_channels=channels,
                kernel_size=15, stride=1, bias=False, padding='same'
            ),
            nn.BatchNorm1d(num_features=channels),
            nn.ELU(),
        )
        self.dec3 = nn.Sequential(
            nn.Conv1d(
                in_channels=2 * channels, out_channels=channels,
                kernel_size=15, stride=1, bias=False, padding='same'
            ),
            nn.BatchNorm1d(num_features=channels),
            nn.ELU(),
            nn.Conv1d(
                in_channels=channels, out_channels=channels,
                kernel_size=15, stride=1, bias=False, padding='same'
            ),
            nn.BatchNorm1d(num_features=channels),
            nn.ELU()
        )
        
        # Stage 4
        self.enc4 = nn.Sequential(
            nn.Conv1d(
                in_channels=channels, out_channels=channels,
                kernel_size=10, stride=1, bias=False, padding='same'
            ),
            nn.BatchNorm1d(num_features=channels),
            nn.ELU()
        )
        self.dec4 = nn.Sequential(
            nn.Conv1d(
                in_channels=2 * channels, out_channels=channels,
                kernel_size=10, stride=1, padding='same', bias=False
            ),
            nn.BatchNorm1d(num_features=channels),
            nn.ELU(),
            nn.Conv1d(
                in_channels=channels, out_channels=channels,
                kernel_size=10, stride=1, bias=False, padding='same'
            ),
            nn.BatchNorm1d(num_features=channels),
            nn.ELU()
        )

        # Stage 5
        self.enc5 = nn.Sequential(
            nn.Conv1d(
                in_channels=channels, out_channels=channels,
                kernel_size=5, stride=1, padding='same', bias=False
            ),
            nn.BatchNorm1d(num_features=channels),
            nn.ELU(),
            nn.Conv1d(
                in_channels=channels, out_channels=channels,
                kernel_size=5, stride=1, bias=False, padding='same'
            ),
            nn.BatchNorm1d(num_features=channels),
            nn.ELU()
        )

        # Center prediction head
        self.center_logit = nn.Conv1d(
            in_channels=channels, out_channels=1,
            kernel_size=7, stride=1, padding='same'
        )
        self.center_logit.bias = bias

        # Duration prediction head
        self.duration_logit = nn.Conv1d(
            in_channels=channels, out_channels=1,
            kernel_size=7, stride=1, padding='same'
        )

    def forward(self, input: Tensor) -> EventOutput:
        # Base
        lvl0 = self.enc0(input)
        # Encoder
        x = self.down(lvl0)
        lvl1 = self.enc1(x) # 1x 

        x = self.down(lvl1) # 2x
        lvl2 = self.enc2(x) # 2x

        x = self.down(lvl2) # 4x
        lvl3 = self.enc3(x) # 4x

        x = self.down(lvl3) # 4x
        lvl4 = self.enc4(x)  # 4x

        x = self.down(lvl4) # 4x
        lvl5 = self.enc5(x) # 4x

        # Decoder
        up4 = self.up(lvl5) # 4x
        x = torch.cat((up4, lvl4), dim=1)
        out4 = self.dec4(x) # 4x

        up3 = self.up(out4) # 4x
        x = torch.cat((up3, lvl3), dim=1)
        out3 = self.dec3(x) # 2x

        up2 = self.up(out3) # 2x
        x = torch.cat((up2, lvl2), dim=1)
        out2 = self.dec2(x) # 2x

        # Outputs
        center_logit = self.center_logit(out2)
        center = self.sigm(center_logit)

        x = self.duration_logit(out2)
        duration = self.sigm(x)

        output = EventOutput(
            center=center,
            duration=duration,
            center_logit=center_logit
        )

        return output
    

class Epoch(nn.Module):
    def __init__(
        self, channels: int = 64, dropout: Optional[float] = None
    ) -> None:
        super().__init__()

        stride = 4
        prior = torch.tensor([1e-1,], dtype=torch.float32)
        bias = nn.parameter.Parameter(
            data=-torch.log((1 - prior) / prior)
        )


        # Utilities
        self.up = nn.Upsample(scale_factor=stride)
        self.sigm = nn.Sigmoid()

        # First patch
        # Stage 0
        self.enc0 = nn.Sequential(
            nn.Conv1d(
                in_channels=1, out_channels=channels // 2,
                stride=1, kernel_size=20, padding='same'
            ),
            nn.ELU(inplace=True)
        )
        self.down = nn.MaxPool1d(kernel_size=stride)

        # Stage 1
        self.enc1 = nn.Sequential(
            nn.Conv1d(
                in_channels=channels // 2, out_channels=channels,
                kernel_size=20, stride=1, bias=False, padding='same'
            ),
            nn.BatchNorm1d(num_features=channels),
            nn.ELU(inplace=True)
        )

        # Stage 2
        self.enc2 = nn.Sequential(
            nn.Conv1d(
                in_channels=channels, out_channels=channels,
                kernel_size=15, stride=1, bias=False, padding='same'
            ),
            nn.BatchNorm1d(num_features=channels),
            nn.ELU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.Conv1d(
                in_channels=2 * channels, out_channels=channels,
                kernel_size=15, stride=1, bias=False, padding='same'
            ),
            nn.BatchNorm1d(num_features=channels),
            nn.ELU(),
            nn.Conv1d(
                in_channels=channels, out_channels=channels,
                kernel_size=15, stride=1, bias=False, padding='same'
            ),
            nn.BatchNorm1d(num_features=channels),
            nn.ELU(),
        )

        # Stage 3
        self.enc3 = nn.Sequential(
            nn.Conv1d(
                in_channels=channels, out_channels=channels,
                kernel_size=15, stride=1, bias=False, padding='same'
            ),
            nn.BatchNorm1d(num_features=channels),
            nn.ELU(),
        )
        self.dec3 = nn.Sequential(
            nn.Conv1d(
                in_channels=2 * channels, out_channels=channels,
                kernel_size=15, stride=1, bias=False, padding='same'
            ),
            nn.BatchNorm1d(num_features=channels),
            nn.ELU(),
            nn.Conv1d(
                in_channels=channels, out_channels=channels,
                kernel_size=15, stride=1, bias=False, padding='same'
            ),
            nn.BatchNorm1d(num_features=channels),
            nn.ELU()
        )
        
        # Stage 4
        self.enc4 = nn.Sequential(
            nn.Conv1d(
                in_channels=channels, out_channels=channels,
                kernel_size=10, stride=1, bias=False, padding='same'
            ),
            nn.BatchNorm1d(num_features=channels),
            nn.ELU()
        )
        self.dec4 = nn.Sequential(
            nn.Conv1d(
                in_channels=2 * channels, out_channels=channels,
                kernel_size=10, stride=1, padding='same', bias=False
            ),
            nn.BatchNorm1d(num_features=channels),
            nn.ELU(),
            nn.Conv1d(
                in_channels=channels, out_channels=channels,
                kernel_size=10, stride=1, bias=False, padding='same'
            ),
            nn.BatchNorm1d(num_features=channels),
            nn.ELU()
        )

        # Stage 5
        self.enc5 = nn.Sequential(
            nn.Conv1d(
                in_channels=channels, out_channels=channels,
                kernel_size=5, stride=1, padding='same', bias=False
            ),
            nn.BatchNorm1d(num_features=channels),
            nn.ELU(),
            nn.Conv1d(
                in_channels=channels, out_channels=channels,
                kernel_size=5, stride=1, bias=False, padding='same'
            ),
            nn.BatchNorm1d(num_features=channels),
            nn.ELU()
        )

        # Center prediction head
        self.center_logit = nn.Conv1d(
            in_channels=channels, out_channels=1,
            kernel_size=7, stride=1, padding='same'
        )

    def forward(self, input: Tensor) -> Tensor:
        # Base
        lvl0 = self.enc0(input)
        # Encoder
        x = self.down(lvl0)
        lvl1 = self.enc1(x) # 1x 

        x = self.down(lvl1) # 2x
        lvl2 = self.enc2(x) # 2x

        x = self.down(lvl2) # 4x
        lvl3 = self.enc3(x) # 4x

        x = self.down(lvl3) # 4x
        lvl4 = self.enc4(x)  # 4x

        x = self.down(lvl4) # 4x
        lvl5 = self.enc5(x) # 4x

        # Decoder
        up4 = self.up(lvl5) # 4x
        x = torch.cat((up4, lvl4), dim=1)
        out4 = self.dec4(x) # 4x

        up3 = self.up(out4) # 4x
        x = torch.cat((up3, lvl3), dim=1)
        out3 = self.dec3(x) # 2x

        up2 = self.up(out3) # 2x
        x = torch.cat((up2, lvl2), dim=1)
        out2 = self.dec2(x) # 2x

        # Outputs
        center_logit = self.center_logit(out2)
        center = self.sigm(center_logit)

        return center
"""
Transformer Model
-----------------
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from numpy.random import RandomState

from ..logging import get_logger
from ..utils.torch import random_method
from ._informer_models.attn import AttentionLayer, FullAttention, ProbAttention
from ._informer_models.decoder import Decoder, DecoderLayer
from ._informer_models.embed import DataEmbedding
from ._informer_models.encoder import ConvLayer, Encoder, EncoderLayer, EncoderStack
from .torch_forecasting_model import PastCovariatesTorchModel

logger = get_logger(__name__)

class Informer(nn.Module):
    def __init__(
        self,
        enc_in,                     # encoder input size
        dec_in,                     # decoder input size
        c_out,                      # output size
        out_len,                    # output length
        factor=5,                   # Probsparse attn factor (defaults to 5)
        d_model=512,                # Dimension of model (defaults to 512)
        n_heads=8,                  # Num of heads (defaults to 8)
        e_layers=3,                 # Num of encoder layers (defaults to 3)
        d_layers=2,                 # Num of decoder layers (defaults to 2)
        d_ff=512,                   # Dimension of fcn (defaults to 2048)
        dropout=0.0,                # The probability of dropout (defaults to 0.0)
        attn="prob",                # Attention used in encoder (defaults to prob). This can be set to prob (informer), full (transformer)
        embed="fixed",              # Time features encoding (defaults to timeF). This can be set to timeF, fixed, learned
        freq="h",                   # Freq for time features encoding (defaults to h). This can be set to s,t,h,d,b,w,m (s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly).You can also use more detailed freq like 15min or 3h
        activation="gelu",
        output_attention=False, 
        distil=True,                # Whether to use distilling in encoder, using this argument means not using distilling (defaults to True)
        mix=True,                   # Whether to use mix attention in generative decoder, using this argument means not using mix attention (defaults to True)
    ):
        super(Informer, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn == "prob" else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        Attn(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=output_attention,
                        ),
                        d_model,
                        n_heads,
                        mix=False,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(e_layers)
            ],
            [ConvLayer(d_model) for l in range(e_layers - 1)] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model),
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        Attn(
                            True,
                            factor,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        d_model,
                        n_heads,
                        mix=mix,
                    ),
                    AttentionLayer(
                        FullAttention(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        d_model,
                        n_heads,
                        mix=False,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(
        self,
        x_enc,
        x_mark_enc,
        x_dec,
        x_mark_dec,
        enc_self_mask=None,
        dec_self_mask=None,
        dec_enc_mask=None,
    ):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(
            dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask
        )
        dec_out = self.projection(dec_out)

        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:, -self.pred_len :, :], attns
        else:
            return dec_out[:, -self.pred_len :, :]  # [B, L, D]

class _InformerModule(nn.Module):
    def __init__(self,
                 input_chunk_length: int,
                 output_chunk_length: int,
                 input_size: int,
                 output_size: int,
                 d_model: int,
                 nhead: int,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 dim_feedforward: int,
                 dropout: float,
                 activation: str,
                 custom_encoder: Optional[nn.Module] = None,
                 custom_decoder: Optional[nn.Module] = None,
                 ):
        super(_InformerModule, self).__init__()

        self.input_size = input_size
        self.target_size = output_size
        self.target_length = output_chunk_length

        self.encoder = nn.Linear(input_size, d_model)
        self.positional_encoding = _PositionalEncoding(d_model, dropout, input_chunk_length)

        # Defining the Transformer module
        self.transformer = nn.Transformer(d_model=d_model,
                                          nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout,
                                          activation=activation,
                                          custom_encoder=custom_encoder,
                                          custom_decoder=custom_decoder)

        self.decoder = nn.Linear(d_model, output_chunk_length * output_size)

    def _create_transformer_inputs(self, data):
        # '_TimeSeriesSequentialDataset' stores time series in the
        # (batch_size, input_chunk_length, input_size) format. PyTorch's nn.Transformer
        # module needs it the (input_chunk_length, batch_size, input_size) format.
        # Therefore, the first two dimensions need to be swapped.
        src = data.permute(1, 0, 2)
        tgt = src[-1:, :, :]

        return src, tgt

    def forward(self, data):
        # Here we create 'src' and 'tgt', the inputs for the encoder and decoder
        # side of the Transformer architecture
        src, tgt = self._create_transformer_inputs(data)
        print(f"src.shape={src.shape}, tgt.shape={tgt.shape}")
        # "math.sqrt(self.input_size)" is a normalization factor
        # see section 3.2.1 in 'Attention is All you Need' by Vaswani et al. (2017)
        src = self.encoder(src) * math.sqrt(self.input_size)
        src = self.positional_encoding(src)

        tgt = self.encoder(tgt) * math.sqrt(self.input_size)
        tgt = self.positional_encoding(tgt)

        x = self.transformer(src=src,
                             tgt=tgt)
        out = self.decoder(x)

        # Here we change the data format
        # from (1, batch_size, output_chunk_length * output_size)
        # to (batch_size, output_chunk_length, output_size)
        predictions = out[0, :, :]
        predictions = predictions.view(-1, self.target_length, self.target_size)
        print(f"predictions.shape={predictions.shape}")
        return predictions
class InformerModel(PastCovariatesTorchModel):
    @random_method
    def __init__(self,
                 input_chunk_length: int,
                 output_chunk_length: int,
                 d_model: int = 64,
                 nhead: int = 4,
                 num_encoder_layers: int = 3,
                 num_decoder_layers: int = 3,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 activation: str = "relu",
                 custom_encoder: Optional[nn.Module] = None,
                 custom_decoder: Optional[nn.Module] = None,
                 random_state: Optional[Union[int, RandomState]] = None,
                 **kwargs):

        kwargs['input_chunk_length'] = input_chunk_length
        kwargs['output_chunk_length'] = output_chunk_length
        super().__init__(**kwargs)

        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.custom_encoder = custom_encoder
        self.custom_decoder = custom_decoder

    def _create_model(self, train_sample: Tuple[torch.Tensor]) -> torch.nn.Module:
        # samples are made of (past_target, past_covariates, future_target)
        input_dim = train_sample[0].shape[1] + (train_sample[1].shape[1] if train_sample[1] is not None else 0)
        output_dim = train_sample[-1].shape[1]

        return _InformerModule(input_chunk_length=self.input_chunk_length,
                                  output_chunk_length=self.output_chunk_length,
                                  input_size=input_dim,
                                  output_size=output_dim,
                                  d_model=self.d_model,
                                  nhead=self.nhead,
                                  num_encoder_layers=self.num_encoder_layers,
                                  num_decoder_layers=self.num_decoder_layers,
                                  dim_feedforward=self.dim_feedforward,
                                  dropout=self.dropout,
                                  activation=self.activation,
                                  custom_encoder=self.custom_encoder,
                                  custom_decoder=self.custom_decoder)
        
        
if __name__ == "__main__":
    
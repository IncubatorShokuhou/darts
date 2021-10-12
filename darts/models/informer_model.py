"""
Transformer Model
-----------------
"""

from typing import Optional, Tuple, Union
import loguru

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
from loguru import logger as loguru_logger

logger = get_logger(__name__)


class Informer(nn.Module):
    def __init__(
        self,
        enc_in=7,  # encoder input size,7
        dec_in=7,  # decoder input size,7
        output_size=1,  # output size, dimension of the output;`c_out` in native Informer
        input_chunk_length=96,  # input length; `seq_len` in native Informer
        label_len=48,  # label(encoding) length;
        output_chunk_length=24,  # output length; `out_len` in native Informer
        factor=5,  # Probsparse attn factor (defaults to 5)
        d_model=512,  # Dimension of model (defaults to 512)
        nhead=8,  # Num of heads (defaults to 8)
        num_encoder_layers=3,  # Num of encoder layers (defaults to 3)
        num_decoder_layers=2,  # Num of decoder layers (defaults to 2)
        dim_feedforward=512,  # Dimension of fcn (defaults to 2048)
        dropout=0.0,  # The probability of dropout (defaults to 0.0)
        attn="prob",  # Attention used in encoder (defaults to prob). This can be set to prob (informer), full (transformer)
        embed="fixed",  # Time features encoding (defaults to timeF). This can be set to timeF, fixed, learned
        freq="h",  # Freq for time features encoding (defaults to h). This can be set to s,t,h,d,b,w,m (s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly).You can also use more detailed freq like 15min or 3h
        activation="gelu",
        output_attention=False,
        distil=True,  # Whether to use distilling in encoder, using this argument means not using distilling (defaults to True)
        mix=True,  # Whether to use mix attention in generative decoder, using this argument means not using mix attention (defaults to True)
    ):
        super(Informer, self).__init__()
        self.output_chunk_length = output_chunk_length
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
                        nhead,
                        mix=False,
                    ),
                    d_model,
                    dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(num_encoder_layers)
            ],
            [ConvLayer(d_model) for l in range(num_encoder_layers - 1)]
            if distil
            else None,
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
                        nhead,
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
                        nhead,
                        mix=False,
                    ),
                    d_model,
                    dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(num_decoder_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, output_size, bias=True)

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
        loguru_logger.debug(f"x_enc.shape={x_enc.shape}")
        loguru_logger.debug(f"x_mark_enc={x_mark_enc.shape}")
        loguru_logger.debug(f"x_dec.shape={x_dec.shape}")
        loguru_logger.debug(f"x_mark_dec={x_mark_dec.shape}")
        

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
            return dec_out[:, -self.output_chunk_length :, :], attns
        else:
            return dec_out[:, -self.output_chunk_length :, :]  # [B, L, D]


class InformerStack(nn.Module):
    def __init__(
        self,
        enc_in,
        dec_in,
        output_size,
        input_chunk_length,
        label_len,
        output_chunk_length,
        factor=5,
        d_model=512,
        nhead=8,
        num_encoder_layers=[3, 2, 1],
        num_decoder_layers=2,
        dim_feedforward=512,
        dropout=0.0,
        attn="prob",
        embed="fixed",
        freq="h",
        activation="gelu",
        output_attention=False,
        distil=True,
        mix=True,
    ):
        super(InformerStack, self).__init__()
        self.output_chunk_length = output_chunk_length
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn == "prob" else FullAttention
        # Encoder

        inp_lens = list(
            range(len(num_encoder_layers))
        )  # [0,1,2,...] you can customize here
        encoders = [
            Encoder(
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
                            nhead,
                            mix=False,
                        ),
                        d_model,
                        dim_feedforward,
                        dropout=dropout,
                        activation=activation,
                    )
                    for l in range(el)
                ],
                [ConvLayer(d_model) for l in range(el - 1)] if distil else None,
                norm_layer=torch.nn.LayerNorm(d_model),
            )
            for el in num_encoder_layers
        ]
        self.encoder = EncoderStack(encoders, inp_lens)
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
                        nhead,
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
                        nhead,
                        mix=False,
                    ),
                    d_model,
                    dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(num_decoder_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, output_size, bias=True)

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
            return dec_out[:, -self.output_chunk_length :, :], attns
        else:
            return dec_out[:, -self.output_chunk_length :, :]  # [B, L, D]


class _InformerModule(nn.Module):
    def __init__(
        self,
        input_chunk_length: int = 96,  # Input sequence length of Informer encoder ; seq_len in native Informer
        output_chunk_length: int = 24,  # 	Prediction sequence length; pred_len in native Informer
        label_len: int = 48,  # Start token length of Informer decoder
        model_name: str = "informer",  # name of model used in Informer. informer or informerstack
        num_encoder_layers: int = 2,  # number of encoder layers; `e_layers` in native Informer
        num_decoder_layers: int = 1,  # num of decoder layers; `d_layers` in native Informer
        s_layers: str = "3,2,1",  # num of stack encoder layers
        enc_in: int = 7,  # Encoder input size
        dec_in: int = 7,  # Decoder input size
        output_size: int = 7,  # Output size
        factor: int = 5,  # Probsparse attn factor
        d_model: int = 512,  # 	Dimension of model
        nhead: int = 8,  # Number of heads in multi-head attention
        dim_feedforward: int = 2048,  # 	Dimension of fcn; `d_ff` in  native Informer
        dropout: float = 0.05,  # The probability of dropout
        attn: str = "prob",  # Attention used in encoder. This can be set to `prob` (informer), `full` (transformer)
        embed: str = "timeF",  # Time features encoding (defaults to timeF). This can be set to `timeF`, `fixed`, `learned`
        freq: str = "h",  # Freq for time features encoding. This can be set to s,t,h,d,b,w,m (s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly).You can also use more detailed freq like 15min or 3h
        activation: str = "gelu",  # Activation function
        output_attention: bool = False,  # Whether to output attention in encoder, using this argument means outputing attention
        distil: bool = True,  # Whether to use distilling in encoder, using this argument means not using distilling
        mix: bool = True,  # Whether to use mix attention in generative decoder, using this argument means not using mix attention
        random_state: Optional[Union[int, RandomState]] = None,
    ):
        super(_InformerModule, self).__init__()

        self.model_name = model_name
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.s_layers = s_layers
        self.enc_in = enc_in
        self.dec_in = dec_in
        self.output_size = output_size
        self.input_chunk_length = input_chunk_length
        self.label_len = label_len
        self.output_chunk_length = output_chunk_length
        self.factor = factor
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.attn = attn
        self.embed = embed
        self.freq = freq
        self.activation = activation
        self.output_attention = output_attention
        self.distil = distil
        self.mix = mix
        self.built_model = self._build_model()

    def _build_model(self):
        model_dict = {
            "informer": Informer,
            "informerstack": InformerStack,
        }
        if self.model_name == "informer" or self.model_name == "informerstack":
            num_encoder_layers = (
                self.num_encoder_layers
                if self.model_name == "informer"
                else self.s_layers
            )
            built_model = model_dict[self.model_name](
                enc_in=self.enc_in,  # encoder input size,7
                dec_in=self.dec_in,  # decoder input size,7
                output_size=self.output_size,  # output size, dimension of the output;`c_out` in native Informer
                input_chunk_length=self.input_chunk_length,  # input length; `seq_len` in native Informer
                label_len=self.label_len,  # label(encoding) length;
                output_chunk_length=self.output_chunk_length,  # output length; `out_len` in native Informer
                factor=self.factor,  # Probsparse attn factor (defaults to 5)
                d_model=self.d_model,  # Dimension of model (defaults to 512)
                nhead=self.nhead,  # Num of heads (defaults to 8)
                num_encoder_layers=num_encoder_layers,  # Num of encoder layers (defaults to 3)
                num_decoder_layers=self.num_decoder_layers,  # Num of decoder layers (defaults to 2)
                dim_feedforward=self.dim_feedforward,  # Dimension of fcn (defaults to 2048)
                dropout=self.dropout,  # The probability of dropout (defaults to 0.0)
                attn=self.attn,  # Attention used in encoder (defaults to prob). This can be set to prob (informer), full (transformer)
                embed=self.embed,  # Time features encoding (defaults to timeF). This can be set to timeF, fixed, learned
                freq=self.freq,  # Freq for time features encoding (defaults to h). This can be set to s,t,h,d,b,w,m (s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly).You can also use more detailed freq like 15min or 3h
                activation=self.activation,
                output_attention=self.output_attention,
                distil=self.distil,  # Whether to use distilling in encoder, using this argument means not using distilling (defaults to True)
                mix=self.mix,
            )

        return built_model

    def _create_informer_inputs(self, data):
        # loguru_logger.info(f"data.shape: {data.shape}") #[32, 30, 2]
        # x_enc.shape=torch.Size([32, 96, 7]), 
        # x_mark_enc=torch.Size([32, 96, 4]), 
        # x_dec.shape=torch.Size([32, 72, 7]), 
        # x_mark_dec=torch.Size([32, 72, 4]),
        
        
        x_enc,x_mark_enc,x_dec,x_mark_dec
        
        
        
        enc_self_mask=None, 
        dec_self_mask=None,
        dec_enc_mask=None
        
        return x_enc,x_mark_enc,x_dec,x_mark_dec,enc_self_mask,dec_self_mask,dec_enc_mask

    def forward(self,        data        ):
        x_enc,x_mark_enc,x_dec,x_mark_dec,enc_self_mask,dec_self_mask,dec_enc_mask = self._create_informer_inputs(data)
        out = self.built_model(x_enc,x_mark_enc,x_dec,x_mark_dec,enc_self_mask,dec_self_mask,dec_enc_mask)
        loguru_logger.debug(f"{out.shape}")

        return out


class InformerModel(PastCovariatesTorchModel):
    @random_method
    def __init__(
        self,
        input_chunk_length: int = 96,  # Input sequence length of Informer encoder ; seq_len in native Informer
        output_chunk_length: int = 24,  # 	Prediction sequence length; pred_len in native Informer
        model_name: str = "informer",  # name of model used in Informer. informer or informerstack; `model` in native Informer. Cannot use `model` because `TorchForecastingModel` have attribute named `model`
        num_encoder_layers: int = 2,  # number of encoder layers; `e_layers` in native Informer
        num_decoder_layers: int = 1,  # num of decoder layers; `d_layers` in native Informer
        s_layers: str = "3,2,1",  # num of stack encoder layers
        enc_in: int = 7,  # Encoder input size. maybe need to be same as train input dim.
        dec_in: int = 7,  # Decoder input size
        output_size: int = 7,  # Output size
        label_len: int = 48,  # Start token length of Informer decoder
        factor: int = 5,  # Probsparse attn factor
        d_model: int = 512,  # 	Dimension of model
        nhead: int = 8,  # Number of heads in multi-head attention
        dim_feedforward: int = 2048,  # 	Dimension of fcn; `d_ff` in  native Informer
        dropout: float = 0.05,  # The probability of dropout
        attn: str = "prob",  # Attention used in encoder. This can be set to `prob` (informer), `full` (transformer)
        embed: str = "timeF",  # Time features encoding (defaults to timeF). This can be set to `timeF`, `fixed`, `learned`
        freq: str = "h",  # Freq for time features encoding. This can be set to s,t,h,d,b,w,m (s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly).You can also use more detailed freq like 15min or 3h
        activation: str = "gelu",  # Activation function
        output_attention: bool = False,  # Whether to output attention in encoder, using this argument means outputing attention
        distil: bool = True,  # Whether to use distilling in encoder, using this argument means not using distilling
        mix: bool = True,  # Whether to use mix attention in generative decoder, using this argument means not using mix attention
        random_state: Optional[Union[int, RandomState]] = None,
        **kwargs,
    ):

        kwargs["input_chunk_length"] = input_chunk_length
        kwargs["output_chunk_length"] = output_chunk_length
        super().__init__(**kwargs)
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length

        self.model_name = model_name
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.s_layers = s_layers
        self.enc_in = enc_in
        self.dec_in = dec_in
        self.output_size = output_size
        self.label_len = label_len
        self.factor = factor
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.attn = attn
        self.embed = embed
        self.freq = freq
        self.activation = activation
        self.output_attention = output_attention
        self.distil = distil
        self.mix = mix
        self.random_state = random_state

    def _create_model(self, train_sample: Tuple[torch.Tensor]) -> torch.nn.Module:
        # samples are made of (past_target, past_covariates, future_target)

        input_dim = train_sample[0].shape[1] + (
            train_sample[1].shape[1] if train_sample[1] is not None else 0
        )
        output_dim = train_sample[-1].shape[1]

        print(f"input_dim={input_dim},output_dim={output_dim}")

        return _InformerModule(
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            label_len=self.label_len,
            model_name=self.model_name,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            s_layers=self.s_layers,
            enc_in=self.enc_in,
            dec_in=self.dec_in,
            output_size=self.output_size,
            factor=self.factor,
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            attn=self.attn,
            embed=self.embed,
            freq=self.freq,
            activation=self.activation,
            output_attention=self.output_attention,
            distil=self.distil,
            mix=self.mix,
            random_state=self.random_state,
        )

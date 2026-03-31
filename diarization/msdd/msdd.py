import json
import os
import tempfile
import wave

from typing import Union

import torch
import torch.nn.functional as F

from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from nemo.collections.asr.modules.msdd_diarizer import MSDD_module
from nemo.collections.asr.parts.utils.speaker_utils import rttm_to_labels
from omegaconf import OmegaConf


def patch_msdd_module_for_mps():
    if getattr(MSDD_module, "_whisper_diarization_mps_patch", False):
        return

    def cosine_similarity(self, scale_weights, ms_avg_embs, _ms_emb_seq):
        cos_dist_seq = self.cos_dist(_ms_emb_seq, ms_avg_embs)
        context_vectors = torch.mul(scale_weights, cos_dist_seq)
        context_vectors = context_vectors.reshape(self.batch_size, self.length, -1)
        context_emb = self.dist_to_emb(context_vectors)
        return context_emb

    def conv_scale_weights(self, ms_avg_embs_perm, ms_emb_seq_single):
        ms_cnn_input_seq = torch.cat([ms_avg_embs_perm, ms_emb_seq_single], dim=2)
        ms_cnn_input_seq = ms_cnn_input_seq.unsqueeze(2).flatten(0, 1)

        conv_out = self.conv_forward(
            ms_cnn_input_seq, conv_module=self.conv[0], bn_module=self.conv_bn[0], first_layer=True
        )
        for conv_idx in range(1, self.conv_repeat + 1):
            conv_out = self.conv_forward(
                conv_input=conv_out,
                conv_module=self.conv[conv_idx],
                bn_module=self.conv_bn[conv_idx],
                first_layer=False,
            )

        lin_input_seq = conv_out.reshape(
            self.batch_size,
            self.length,
            self.cnn_output_ch * self.emb_dim,
        )
        hidden_seq = self.conv_to_linear(lin_input_seq)
        hidden_seq = self.dropout(F.leaky_relu(hidden_seq))
        scale_weights = self.softmax(self.linear_to_weights(hidden_seq))
        scale_weights = scale_weights.unsqueeze(3).expand(-1, -1, -1, self.num_spks)
        return scale_weights

    MSDD_module.cosine_similarity = cosine_similarity
    MSDD_module.conv_scale_weights = conv_scale_weights
    MSDD_module._whisper_diarization_mps_patch = True


class MSDDDiarizer:
    def __init__(self, device: Union[str, torch.device]):
        if str(device) == "mps":
            # NeMo's MSDD implementation uses Tensor.view() on non-contiguous tensors,
            # which fails on MPS. reshape() preserves the same semantics here.
            patch_msdd_module_for_mps()
        self.model: NeuralDiarizer = NeuralDiarizer(cfg=create_config()).to(device)

    def diarize(self, audio: torch.Tensor):
        with tempfile.TemporaryDirectory() as temp_path:
            pcm = (audio.cpu().numpy() * 32768).clip(-32768, 32767).astype("int16")
            with wave.open(os.path.join(temp_path, "mono_file.wav"), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(pcm.tobytes())

            manifest_path = os.path.join(temp_path, "manifest.json")
            meta = {
                "audio_filepath": os.path.join(temp_path, "mono_file.wav"),
                "offset": 0,
                "duration": None,
                "label": "infer",
                "text": "-",
                "rttm_filepath": None,
                "uem_filepath": None,
            }

            with open(manifest_path, "w") as f:
                json.dump(meta, f)

            self.model._initialize_configs(
                manifest_path=manifest_path,
                max_speakers=8,
                num_speakers=None,
                tmpdir=temp_path,
                batch_size=24,
                num_workers=0,
                verbose=True,
            )
            self.model.clustering_embedding.clus_diar_model._diarizer_params.out_dir = temp_path
            self.model.clustering_embedding.clus_diar_model._diarizer_params.manifest_filepath = (
                manifest_path
            )
            self.model.msdd_model.cfg.test_ds.manifest_filepath = manifest_path
            self.model.diarize()

            pred_labels_clus = rttm_to_labels(
                os.path.join(temp_path, "pred_rttms", "mono_file.rttm")
            )

            labels = []
            for label in pred_labels_clus:
                start, end, speaker = label.split()
                start, end = float(start), float(end)
                start, end = int(start * 1000), int(end * 1000)
                labels.append((start, end, int(speaker.split("_")[1])))

            labels = sorted(labels, key=lambda x: x[0])

        return labels


def create_config():
    config = OmegaConf.load(os.path.join(os.path.dirname(__file__), "diar_infer_telephonic.yaml"))
    pretrained_vad = "vad_multilingual_marblenet"
    pretrained_speaker_model = "titanet_large"

    config.diarizer.out_dir = None
    config.diarizer.manifest_filepath = None
    config.diarizer.speaker_embeddings.model_path = pretrained_speaker_model
    config.diarizer.oracle_vad = False  # compute VAD provided with model_path to vad config
    config.diarizer.clustering.parameters.oracle_num_speakers = False

    # Here, we use our in-house pretrained NeMo VAD model
    config.diarizer.vad.model_path = pretrained_vad
    config.diarizer.vad.parameters.onset = 0.8
    config.diarizer.vad.parameters.offset = 0.6
    config.diarizer.vad.parameters.pad_offset = -0.05
    config.diarizer.msdd_model.model_path = (
        "diar_msdd_telephonic"  # Telephonic speaker diarization model
    )

    return config

import sys
sys.path.append("..")
import torch
import torch.nn as nn
import torch.nn.functional as F
print(torch.__version__)
from spikingjelly.activation_based import surrogate
from spikingjelly.activation_based import neuron
import numpy as np

# tau = 10.0 # beta = 1 - 1/tau
backend = "torch"
detach_reset=True
# common_thr = 1.0
# attn_thr = common_thr / 4
from spikingjelly.activation_based.neuron import LIFNode, BaseNode 

# Attempt to import predictor-related functions from snn_modules.py
# This path might need adjustment based on your project structure and PYTHONPATH
try:
    from .spike_update_predictor import load_predictor, predict_spike
except ImportError:
    # Fallback or error if direct import fails.
    # You might need to add SpikeBERT's parent directory to PYTHONPATH
    # or adjust the import path.
    print("Warning: Could not directly import from SpikeBERT.implicit_bert.modules.snn_modules.")
    print("Please ensure the SpikeBERT package is correctly installed or paths are set up.")
    # As a placeholder if the import fails during generation, define dummy functions.
    # In a real scenario, this import MUST work.
    


# Try to import pandas, as it's used by the reference prepare_features logic
try:
    import pandas as pd
except ImportError:
    print("Warning: pandas library not found. PredictiveLIFNode's feature preparation might fail.")
    pd = None # Allow class definition but it will fail at runtime if pd is used.


class PredictiveLIFNode(LIFNode):
    def __init__(self, 
                 predictor_path: str, 
                 tau: float = 2., 
                 decay_input: bool = True, 
                 v_threshold: float = 1.,
                 v_reset: float = 0., 
                 surrogate_function=surrogate.ATan(), # Using ATan like in new_spikformer
                 detach_reset: bool = False, 
                 step_mode='m', 
                 backend='torch',  # Predictive logic currently best suited for 'torch' backend
                 store_v_seq: bool = False,
                 max_history_len: int = 10,
                 predictor_output_scale: float = 1.0): # Scale for the predicted value
        
        super().__init__(tau=tau, decay_input=decay_input, v_threshold=v_threshold, 
                         v_reset=v_reset, surrogate_function=surrogate_function,
                         detach_reset=detach_reset, step_mode=step_mode, backend=backend, 
                         store_v_seq=store_v_seq)

        if backend == 'cupy' and step_mode == 'm':
            print("Warning: PredictiveLIFNode with backend='cupy' and step_mode='m' will use a PyTorch-based loop for prediction logic, potentially impacting CuPy performance benefits.")

        self.predictor_path = predictor_path
        self.max_history_len = max_history_len
        self.predictor_output_scale = predictor_output_scale

        self.spike_predictor, self.scaler = load_predictor(self.predictor_path)
        self._predict_spike_func = predict_spike
        
        if self.spike_predictor is None:
            print(f"Warning: Spike predictor model could not be loaded from {self.predictor_path}. Predictions will be zero.")

        # Initialize history for predictor features
        self._reset_predictor_history()
        self.current_train_loss = 0.0

    def _reset_predictor_history(self):
        self.history = {
            'prev_spike_rate': 0.0,
            'mean_voltage_list': [],
            'voltage_threshold_ratio_list': [],
            'cumulative_spike_count': 0.0,
            'elements_processed_count': 0,
            'predicted_update_rate': 0.0, # Stores the last output of the predictor itself
        }

    def update_train_loss(self, loss: float):
        """Allows external update of the training loss for feature preparation."""
        self.current_train_loss = loss

    def _calculate_prediction_from_current_history(self) -> float:
        
        # snn_modules.py 에서 참조한 feature_names (고정값으로 사용)
        feature_names = [
            'prev_spike_rate',
            'mean_voltage',
            'voltage_threshold_ratio',
            'cumulative_spikes',
            'predicted_update_rate',
            'train_loss'
        ]

        # Construct features from self.history, matching feature_names order
        raw_features_values = []
        
        # Safely calculate mean for potentially empty lists
        mean_volt_list = self.history['mean_voltage_list']
        mean_volt_ratio_list = self.history['voltage_threshold_ratio_list']

        feat_mean_voltage = np.mean(mean_volt_list) if mean_volt_list else 0.0
        feat_voltage_threshold_ratio = np.mean(mean_volt_ratio_list) if mean_volt_ratio_list else 0.0
        
        feat_cumulative_spikes = (self.history['cumulative_spike_count'] / self.history['elements_processed_count']) \
                                 if self.history['elements_processed_count'] > 0 else 0.0

        for name in feature_names: # self.feature_names 대신 로컬 feature_names 사용
            if name == 'prev_spike_rate': raw_features_values.append(self.history['prev_spike_rate'])
            elif name == 'mean_voltage': raw_features_values.append(abs(feat_mean_voltage))
            elif name == 'voltage_threshold_ratio': raw_features_values.append(abs(feat_voltage_threshold_ratio))
            elif name == 'cumulative_spikes': raw_features_values.append(feat_cumulative_spikes)
            elif name == 'predicted_update_rate': raw_features_values.append(self.history['predicted_update_rate'])
            elif name == 'train_loss': raw_features_values.append(self.current_train_loss)
            else: 
                print(f"Warning: Unknown feature name '{name}' in feature_names. Using 0.0.")
                raw_features_values.append(0.0)
        
        features_np = np.array([raw_features_values], dtype=float)
        features_df = pd.DataFrame(features_np, columns=feature_names) # self.feature_names 대신 로컬 feature_names 사용
        #print(f"[PredictiveLIFNode DEBUG] Features for prediction:\n{features_df}")

        predicted_rate_scalar = self._predict_spike_func(self.spike_predictor, self.scaler, features_df)
        #print(f"[PredictiveLIFNode DEBUG] Raw predicted rate from predictor: {predicted_rate_scalar}")
        
        # Store the prediction for the next step's 'predicted_update_rate' feature
        self.history['predicted_update_rate'] = float(predicted_rate_scalar) if not np.isnan(predicted_rate_scalar) else 0.0
        
        final_prediction = float(np.clip(predicted_rate_scalar, 0.0, 1.0))
        #print(f"[PredictiveLIFNode DEBUG] Clipped prediction (0.0 to 1.0): {final_prediction}")
        return final_prediction


    def _update_history_after_step_core(self, v_for_stats: torch.Tensor, spike_this_step: torch.Tensor):
        # v_for_stats is the voltage snapshot to record (e.g., after charge, before reset)
        mean_v_val = torch.mean(v_for_stats.detach()).cpu().item()
        self.history['mean_voltage_list'].append(mean_v_val)
        if len(self.history['mean_voltage_list']) > self.max_history_len:
            self.history['mean_voltage_list'].pop(0)
        
        v_thresh_ratio_val = mean_v_val / self.v_threshold # self.v_threshold is float
        self.history['voltage_threshold_ratio_list'].append(v_thresh_ratio_val)
        if len(self.history['voltage_threshold_ratio_list']) > self.max_history_len:
            self.history['voltage_threshold_ratio_list'].pop(0)

        # Update spike-related history (using spikes from the current step)
        detached_spike = spike_this_step.detach()
        self.history['prev_spike_rate'] = torch.mean(detached_spike).cpu().item() # For *next* step's prediction
        self.history['cumulative_spike_count'] += torch.sum(detached_spike).cpu().item()
        self.history['elements_processed_count'] += detached_spike.numel()

    def _perform_single_step_dynamics(self, x_modified_for_step: torch.Tensor):
        """
        Encapsulates the core LIF dynamics for a single time step using x_modified_for_step.
        This method assumes self.v is already a tensor and correctly initialized.
        It updates self.v and returns the spike.
        """
        # 1. Neuronal Charge (using the modified input)
        if self.decay_input:
            if self.v_reset is None or self.v_reset == 0.:
                self.v = self.v + (x_modified_for_step - self.v) / self.tau
            else:
                self.v = self.v + (x_modified_for_step - (self.v - self.v_reset)) / self.tau
        else:
            if self.v_reset is None or self.v_reset == 0.:
                self.v = self.v * (1. - 1. / self.tau) + x_modified_for_step
            else:
                self.v = self.v - (self.v - self.v_reset) / self.tau + x_modified_for_step
        
        v_after_charge = self.v.clone() # Snapshot for history

        # 2. Neuronal Fire
        spike = self.neuronal_fire() # From BaseNode, uses self.v

        # 3. Neuronal Reset
        self.neuronal_reset(spike) # From BaseNode, updates self.v

        # 4. Update history (for the *next* step's prediction)
        self._update_history_after_step_core(v_after_charge, spike)
        
        return spike

    def single_step_forward(self, x: torch.Tensor):
        # Ensure self.v is a tensor and on the same device as x
        self.v_float_to_tensor(x)

        # 1. Calculate prediction based on current history (reflects state after previous step)
        prediction_scalar = self._calculate_prediction_from_current_history()
        scaled_prediction = prediction_scalar * self.predictor_output_scale
        #print(f"[PredictiveLIFNode DEBUG] Scaled prediction for input current: {scaled_prediction} (raw: {prediction_scalar}, scale: {self.predictor_output_scale})")
        predicted_input_current = torch.full_like(x, float(scaled_prediction), device=x.device)

        # 2. Modified input for LIF dynamics for this single step
        modified_x_this_step = x + predicted_input_current
        
        # 3. Perform LIF dynamics for this step
        spike = self._perform_single_step_dynamics(modified_x_this_step)
        
        return spike

    def multi_step_forward(self, x_seq: torch.Tensor):
        """
        Custom multi-step forward for PredictiveLIFNode.
        Processes a sequence of inputs, applying prediction logic at each step.
        x_seq.shape = [T, B, N, ...] (N is number of neurons, ... are other dims)
        """
        # Ensure self.v is a tensor and correctly initialized based on the first time step's input shape
        self.v_float_to_tensor(x_seq[0])

        T = x_seq.shape[0]
        y_seq = []
        
        if self.store_v_seq:
            v_seq_list = []

        for t in range(T):
            # 1. Calculate prediction based on current history (reflects state after previous step t-1)
            prediction_scalar_t = self._calculate_prediction_from_current_history()
            scaled_prediction_t = prediction_scalar_t * self.predictor_output_scale
            #print(f"[PredictiveLIFNode DEBUG] Step {t}: Scaled prediction for input current: {scaled_prediction_t} (raw: {prediction_scalar_t}, scale: {self.predictor_output_scale})")
            predicted_input_current_t = torch.full_like(x_seq[t], float(scaled_prediction_t), device=x_seq.device)

            # 2. Modified input for LIF dynamics for current step t
            modified_x_t = x_seq[t] + predicted_input_current_t
            
            # 3. Perform LIF dynamics for step t using _perform_single_step_dynamics
            # This updates self.v, self.history, and returns the spike for step t
            spike_t = self._perform_single_step_dynamics(modified_x_t)
            
            y_seq.append(spike_t)
            if self.store_v_seq:
                v_seq_list.append(self.v.clone()) # self.v is already updated by _perform_single_step_dynamics

        if self.store_v_seq:
            self.v_seq = torch.stack(v_seq_list) # Store the collected v_seq

        return torch.stack(y_seq)

    def forward(self, x: torch.Tensor):
        if self.step_mode == 's':
            return self.single_step_forward(x)
        elif self.step_mode == 'm':
            # Call our custom multi_step_forward
            return self.multi_step_forward(x)
        else:
            raise ValueError(f"Unsupported step_mode: {self.step_mode}")

    def reset(self):
        super().reset() # Calls LIFNode.reset -> BaseNode.reset (resets self.v, self.v_seq)
        self._reset_predictor_history()
        # self.current_train_loss is typically managed per epoch/batch, not reset with neuron state.

class spiking_self_attention(nn.Module):
    def __init__(self, length, tau, common_thr, dim, heads=8, qkv_bias=False, qk_scale=0.25):
        super().__init__()
        assert dim % heads == 0, f"dim {dim} should be divided by num_heads {heads}."

        self.dim = dim
        self.heads = heads
        self.qk_scale = qk_scale

        self.q_m = nn.Linear(dim, dim)
        self.q_ln = nn.LayerNorm(dim)
        # self.q_lif = neuron.LIFNode(tau=tau, step_mode='m', detach_reset=detach_reset, surrogate_function=surrogate.ATan(), v_threshold=common_thr, backend=backend)
        self.q_lif = PredictiveLIFNode(tau=tau, detach_reset=detach_reset, predictor_path="./model/spike_predictor_final.pth", v_threshold=common_thr, backend=backend)


        self.k_m = nn.Linear(dim, dim)
        self.k_ln = nn.LayerNorm(dim)
        # self.k_lif = neuron.LIFNode(tau=tau, step_mode='m', detach_reset=detach_reset, surrogate_function=surrogate.ATan(), v_threshold=common_thr, backend=backend)
        self.k_lif = PredictiveLIFNode(tau=tau, detach_reset=detach_reset, predictor_path="./model/spike_predictor_final.pth", v_threshold=common_thr, backend=backend)

        self.v_m = nn.Linear(dim, dim)
        self.v_ln = nn.LayerNorm(dim)
        # self.v_lif = neuron.LIFNode(tau=tau, step_mode='m', detach_reset=detach_reset, surrogate_function=surrogate.ATan(), v_threshold=common_thr, backend=backend)
        self.v_lif = PredictiveLIFNode(tau=tau, detach_reset=detach_reset, predictor_path="./model/spike_predictor_final.pth", v_threshold=common_thr, backend=backend)

        # self.attn_lif = neuron.LIFNode(tau=tau, step_mode='m', detach_reset=detach_reset, surrogate_function=surrogate.ATan(), v_threshold=common_thr/2, backend=backend)
        self.attn_lif = PredictiveLIFNode(tau=tau, detach_reset=detach_reset, predictor_path="./model/spike_predictor_final.pth", v_threshold=common_thr/2, backend=backend)

        self.last_m = nn.Linear(dim, dim)
        self.last_ln = nn.LayerNorm(dim)
        # self.last_lif = neuron.LIFNode(tau=tau, step_mode='m', detach_reset=detach_reset, surrogate_function=surrogate.ATan(), v_threshold=common_thr, backend=backend)
        self.last_lif = PredictiveLIFNode(tau=tau, detach_reset=detach_reset, predictor_path="./model/spike_predictor_final.pth", v_threshold=common_thr, backend=backend)

    def forward(self, x):# B T L D
        x = x.transpose(0, 1) # T B L D

        T, B, L, D = x.shape
        x_for_qkv = x.flatten(0, 1) # TB L D

        q_m_out = self.q_m(x_for_qkv) # TB L D
        q_m_out = self.q_ln(q_m_out).reshape(T, B, L, D).contiguous()
        q_m_out = self.q_lif(q_m_out)
        q = q_m_out.reshape(T, B, L, self.heads, D // self.heads).permute(0, 1, 3, 2, 4).contiguous()

        k_m_out = self.k_m(x_for_qkv)
        k_m_out = self.k_ln(k_m_out).reshape(T, B, L, D).contiguous()
        k_m_out = self.k_lif(k_m_out)
        k = k_m_out.reshape(T, B, L, self.heads, D // self.heads).permute(0, 1, 3, 2, 4).contiguous()

        v_m_out = self.v_m(x_for_qkv)
        v_m_out = self.v_ln(v_m_out).reshape(T, B, L, D).contiguous()
        v_m_out = self.v_lif(v_m_out)
        v = v_m_out.reshape(T, B, L, self.heads, D // self.heads).permute(0, 1, 3, 2, 4).contiguous()

        attn = (q @ k.transpose(-2, -1))
        # print(attn.shape)
        x = (attn @ v) * self.qk_scale  # x_shape: T * B * heads * L * //heads
        # print(x.shape)

        x = x.transpose(2, 3).reshape(T, B, L, D).contiguous()
        # print(x.shape)
        x = self.attn_lif(x)
        
        x = x.flatten(0, 1)
        # print(x.shape)
        x = self.last_m(x)
        x = self.last_ln(x)
        x = self.last_lif(x.reshape(T, B, L, D).contiguous())

        x = x.transpose(0, 1) # B T L D
        return x


class mlp(nn.Module):
    def __init__(self, length, tau, common_thr, in_features, hidden_features=None, out_features=None, ):
        super().__init__()
        # self.length = length
        out_features = out_features or in_features
        hidden_features = hidden_features
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.ln1 = nn.LayerNorm(hidden_features)
        # self.lif1 = neuron.LIFNode(tau=tau, step_mode='m', detach_reset=detach_reset, surrogate_function=surrogate.ATan(), v_threshold=common_thr, backend=backend)
        self.lif1 = PredictiveLIFNode(tau=tau, detach_reset=detach_reset, predictor_path="./model/spike_predictor_final.pth", v_threshold=common_thr, backend=backend)

        self.fc2 = nn.Linear(hidden_features, out_features)
        self.ln2 = nn.LayerNorm(out_features)
        # self.lif2 = neuron.LIFNode(tau=tau, step_mode='m', detach_reset=detach_reset, surrogate_function=surrogate.ATan(), v_threshold=common_thr, backend=backend)
        self.lif2 = PredictiveLIFNode(tau=tau, detach_reset=detach_reset, predictor_path="./model/spike_predictor_final.pth", v_threshold=common_thr, backend=backend)

    def forward(self, x):
        # B T L D
        x = x.transpose(0, 1) # T B L D
        T, B, L, D_in = x.shape # D_in is self.in_features
        x = x.flatten(0, 1) # Shape: (T*B, L, D_in)
        
        # After fc1, last dim becomes self.hidden_features
        x_fc1_out = self.fc1(x) # Shape: (T*B, L, self.hidden_features)
        x_ln1_out = self.ln1(x_fc1_out) # Shape: (T*B, L, self.hidden_features)
        # Reshape to (T, B, L, self.hidden_features) before LIF
        x_reshaped1 = x_ln1_out.reshape(T, B, L, self.hidden_features).contiguous()
        x = self.lif1(x_reshaped1)
        
        # Flatten again for fc2
        x = x.flatten(0, 1) # Shape: (T*B, L, self.hidden_features)
        
        # After fc2, last dim becomes self.out_features
        x_fc2_out = self.fc2(x) # Shape: (T*B, L, self.out_features)
        x_ln2_out = self.ln2(x_fc2_out) # Shape: (T*B, L, self.out_features)
        # Reshape to (T, B, L, self.out_features) before LIF
        x_reshaped2 = x_ln2_out.reshape(T, B, L, self.out_features).contiguous()
        x = self.lif2(x_reshaped2)
        
        x = x.transpose(0, 1) # B T L D (actually B T L self.out_features)
        return x


class block(nn.Module):
    def __init__(self, length, tau, common_thr, dim, heads=8, qkv_bias=False, qk_scale=0.125):
        super().__init__()
        self.attn = spiking_self_attention(length=length, tau=tau, common_thr=common_thr, dim=dim, heads=heads, qkv_bias=qkv_bias, qk_scale=qk_scale)
        self.mlp = mlp(length=length, tau=tau, common_thr=common_thr, in_features=dim, hidden_features=dim*4, out_features=dim)

    def forward(self, x):
        # B T L D
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class transform(nn.Module):
    def __init__(self, dim, length):
        super(transform, self).__init__()
        self.fc = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)
    def forward(self, x):
        x = self.fc(x)
        x = self.ln(x)
        return x


class new_spikformer_legacy(nn.Module):
    def __init__(self, depths, length, tau, common_thr, dim, T, vocab_size = 28996, num_classes=2, heads=8, qkv_bias=False, qk_scale=0.125, mode="train"):
        super().__init__()
        self.mode = mode
        self.atan = surrogate.ATan()
        self.T = T
        self.emb = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([block(
            length=length, tau=tau, common_thr=common_thr, dim=dim, heads=heads, qkv_bias=qkv_bias, qk_scale=qk_scale
        ) for _ in range(depths)])
        self.last_ln = nn.LayerNorm(dim)

        self.transforms = nn.ModuleList([
            transform(dim, length) for _ in range(depths)
        ])
        if mode != "pre_distill":
            self.classifier = nn.Linear(dim, num_classes) if num_classes > 0 else nn.Identity()
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    def update_predictor_train_loss(self, current_loss: float):
        """Updates the 'current_train_loss' attribute in all PredictiveLIFNode instances."""
        for module in self.modules():
            if isinstance(module, PredictiveLIFNode):
                module.update_train_loss(current_loss)

    def forward(self, x):
        # B L D
        x = self.emb(x)
        # print(x.shape)
        x = x.repeat(tuple([self.T] + torch.ones(len(x.size()), dtype=int).tolist())) # T B L D
        x = x.transpose(0, 1) # B T L D
        x = self.atan(x)
        representations = []
        for i, blk in enumerate(self.blocks):
            x = blk(x) # B T L D
            representations.append(self.transforms[i](x.mean(1))) # B * L * D
            # last step
            # representations.append(self.transforms[i](x[:,-1,:,:])) # B * L * D
        # B T L D
        x = self.last_ln(x)
        # B T L D
        x = x.mean(2)
        if self.mode != "pre_distill":
            x = self.classifier(x)
        # x: B T D
        return representations, x
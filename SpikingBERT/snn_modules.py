# Identifiers will be add once the code is made public.

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pickle
import sys
import os
from .spike_update_predictor import SpikeUpdatePredictor, load_predictor, predict_spike
sys.path.append('../')
from implicit_bert.modules.optimizations import VariationalHidDropout2d, weight_spectral_norm
import random
import pandas as pd

import warnings
warnings.filterwarnings('ignore')


class SNNFuncMultiLayer(nn.Module):

    def __init__(self, network_s_list, network_x, vth, fb_num=1):
        # network_s_list is a list of networks, the last fb_num ones are the feedback while previous are feed-forward
        super(SNNFuncMultiLayer, self).__init__()
        self.network_s_list = network_s_list
        self.network_x = network_x
        self.vth = torch.tensor(vth, requires_grad=False)
        self.fb_num = fb_num

    def snn_forward(self, x, time_step, output_type='normal', input_type='constant'):
        pass

    # This function is primarily used for training. It leverages the ASR as defined in the paper.
    def equivalent_func_per_layer_bert_spiking_specific(self, a, x, segment_ids, num, attention_mask = None, is_attn = 0, op = 'train'):
        fac = 1.
        min_val = 0
        max_val = 1
        avg_list_prev = []
        avg_list_prev.append(a)
        j = 0
        for i in range(num, len(self.network_s_list) - 1):
            # Comment out this break for feedback
            break
            if i % 6 == 0:
                # Key Layer
                a = torch.clamp((self.network_s_list[i](avg_list_prev[j])) / self.vth, min_val, max_val)
            elif i % 6 == 1:
                # Value Layer
                a = torch.clamp((self.network_s_list[i](avg_list_prev[j-1])) / self.vth, min_val, max_val)
            elif i % 6 == 2:
                a, attn = (self.network_s_list[i](avg_list_prev[j-2], avg_list_prev[j-1], avg_list_prev[j], attention_mask))
                a = torch.clamp((a) / (self.vth), min_val, max_val)
                #if is_attn > 0:
                #    if i + 1 == is_attn:
                #        return attn
            elif i % 6 == 3:
                a = torch.clamp((self.network_s_list[i](avg_list_prev[j], avg_list_prev[j-3])) / self.vth, min_val, max_val)
            elif i % 6 == 5:
                a = torch.clamp((self.network_s_list[i](avg_list_prev[j], avg_list_prev[j-1])) / self.vth, min_val, max_val)
            else:
                a = torch.clamp((self.network_s_list[i](avg_list_prev[j])) / self.vth, min_val, max_val)
            avg_list_prev.append(a)
            j += 1

        # Uncomment this to use feedback
        #a = torch.clamp((self.network_s_list[-1](a) + self.network_x(x, segment_ids)) / self.vth, min_val, max_val)

        # No feedback
        a = torch.clamp((self.network_x(x, segment_ids)) / self.vth, min_val, max_val)

        #a = self.network_s_list[-1](a) + self.network_x(x, segment_ids)
        avg_list = []
        avg_list.append(a)
        if is_attn:
            num = 25
        for i in range(num):
            if i % 6 == 0:
                # Key Layer
                a = torch.clamp((self.network_s_list[i](avg_list[i])) / (self.vth), min_val, max_val)
            elif i % 6 == 1:
                # Value Layer
                a = torch.clamp((self.network_s_list[i](avg_list[i-1])) / (self.vth), min_val, max_val)
            elif i % 6 == 2:
                # Attention layer
                a, attn = (self.network_s_list[i](avg_list[i-2], avg_list[i-1], avg_list[i], attention_mask))
                a = torch.clamp((a) / (self.vth), min_val, max_val)
                if is_attn > 0:
                    if int(i/6) + 1 == is_attn:
                        return attn
            elif i % 6 == 3:
                # IL1
                a = torch.clamp((self.network_s_list[i](avg_list[i], avg_list[i-3])) / self.vth, min_val, max_val)
            elif i % 6 == 5:
                # Output
                a = torch.clamp((self.network_s_list[i](avg_list[i], avg_list[i-1])) / self.vth, min_val, max_val)
            else:
                a = torch.clamp((self.network_s_list[i](avg_list[i])) / self.vth, min_val, max_val)
            avg_list.append(a)
        return a

    def forward(self, x, time_step):
        return self.snn_forward(x, time_step)

    def copy(self, target):
        for i in range(len(self.network_s_list)):
            self.network_s_list[i].copy(target.network_s_list[i])
        self.network_x.copy(target.network_x)


# Spike creation and flow is defined in this class
class SNNBERTSpikingLIFFuncMultiLayer(SNNFuncMultiLayer):

    def __init__(self, network_s_list, network_x, vth, leaky, fb_num=1, predictor_path='spike_predictor_final.pth'):
        super(SNNBERTSpikingLIFFuncMultiLayer, self).__init__(network_s_list, network_x, vth, fb_num)
        self.leaky = torch.tensor(leaky, requires_grad=False)
        
        # 특성 이름 정의
        self.feature_names = [
            'prev_spike_rate',
            'mean_voltage',
            'voltage_threshold_ratio',
            'cumulative_spikes',
            'predicted_update_rate',
            'train_loss'
        ]
        
        print(f"\nInitializing SNNBERTSpikingLIFFuncMultiLayer...")
        print(f"Loading predictor from: {predictor_path}")
        
        if not os.path.isabs(predictor_path):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            predictor_path = os.path.join(current_dir, predictor_path)
            
        self.spike_predictor, self.scaler = load_predictor(predictor_path)
        
        self.layer_history = {}
        
        # 로깅 설정 추가
        self.log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'logs', 'spike_features')
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 성능 메트릭 저장
        self.current_loss = None
        self.current_accuracy = None
        
        # 모델 상태 초기화
        #self.spike_predictor = None
        self.layer_history = {}
        
        # 추론 최적화를 위한 설정
        self.eval()  # 평가 모드로 설정
        
    def prepare_features(self, layer_idx, u, s=None):
        """예측을 위한 특성 준비 - CPU 변환 처리 추가"""
        try:
            if layer_idx not in self.layer_history:
                self.layer_history[layer_idx] = {
                    'prev_spike_rate': 0.0,
                    'mean_voltage': [],
                    'cumulative_spikes': 0,
                    'voltage_threshold_ratio': [],
                    'predicted_update_rate': 0.0,
                    'train_loss': 0.0,
                    'spike_count': 0,
                }
            
            history = self.layer_history[layer_idx]
            


            # GPU 텐서를 CPU로 이동 후 계산
            if not torch.isnan(torch.mean(u)):
                current_voltage = torch.mean(u).cpu().item()
                voltage_threshold_ratio = current_voltage / self.vth.cpu().item()
                
            # 히스토리 업데이트
                history['mean_voltage'].append(current_voltage)
                history['mean_voltage'] = history['mean_voltage'][-10:]
                history['voltage_threshold_ratio'].append(voltage_threshold_ratio)
                history['voltage_threshold_ratio'] = history['voltage_threshold_ratio'][-10:]
            

            history['train_loss'] = self.current_loss if self.current_loss else 0.0
            
            
            # 스파이크 계산 시 CPU로 이동
            if s is not None:
                current_spike_rate = torch.mean(s).cpu().item()
                history['prev_spike_rate'] = current_spike_rate
                history['spike_count'] += torch.sum(s).cpu().item()
                total_elements = s.numel() * (len(history['mean_voltage']))
                history['cumulative_spikes'] = history['spike_count'] / max(1, total_elements)
            
            # NumPy 배열로 변환 전에 모든 값이 CPU에 있는지 확인
            features = [
                history['prev_spike_rate'],
                abs(np.mean(history['mean_voltage'])),
                abs(np.mean(history['voltage_threshold_ratio'])),
                history['cumulative_spikes'],
                history['predicted_update_rate'],
                history['train_loss']
            ]
            
            return np.array([features])
            
        except Exception as e:
            print(f"Error preparing features for layer {layer_idx}: {str(e)}")
            return np.array([[0.0] * len(self.feature_names)])

    def predict_spike_update(self, layer_idx, u, s=None):
        """스파이크 업데이트 예측"""
        if self.spike_predictor is None:
            return 1.0
            
        try:
            # 특성 준비 (CPU에서 처리)
            features = self.prepare_features(layer_idx, u, s)
            features_df = pd.DataFrame(features, columns=self.feature_names)
            
            # NaN 체크 및 처리
            
            #if self.scaler is not None:
            #    features = self.scaler.transform(features_df)
            
            # GPU 메모리 최적화를 위한 처리
            result = predict_spike(self.spike_predictor, self.scaler, features)
            
            # 예측된 업데이트 비율 저장
            if layer_idx in self.layer_history:
                self.layer_history[layer_idx]['predicted_update_rate'] = float(result) if not np.isnan(result) else 0.0
            
            return np.clip(result, 0.0, 1.0)
            
        except Exception as e:
            print(f"Error in prediction for layer {layer_idx}: {str(e)}")
            return 1.0

    def update_metrics(self, loss, accuracy):
        """현재 성능 메트릭 업데이트"""
        self.current_loss = loss
        self.current_accuracy = accuracy
        
        
    def snn_forward(self, x, segment_ids, time_step, flag_num=None, output_type='normal', input_type='constant', attention_mask=None, train_loss=None):
        # GPU 메모리 최적화
        torch.cuda.empty_cache()  # 시작 전 GPU 캐시 정리
    
        # 학습 손실이 제공된 경우 메트릭 업데이트
        
        self.update_metrics(train_loss, None)
        
        # 중간 결과를 저장하는 리스트들을 CPU에 유지
        u_list = []
        s_list = []
        attn_list = []
        fac = 1.0
        if input_type == 'constant':
            x1 = self.network_x(x, segment_ids)
            x1 = x1.detach()  # 그래디언트 히스토리 제거
        
        # 첫 번째 레이어 처리
        u1 = x1
        s1 = (u1 >= self.vth).float()
        u1 = u1 - self.vth * s1
        u1 = u1 * self.leaky
        
        # 각 레이어마다 누적 스파이크 초기화
        if not hasattr(self, 'cumulative_spikes'):
            self.cumulative_spikes = {}
        
        # 첫 번째 레이어 로깅
        #self._init_logging()
        #self._log_features(0, 0, u1, s1, None, 0.0, train_loss)
        
        u_list.append(u1)
        s_list.append(s1)
        
        # 나머지 레이어 초기 처리
        for i in range(len(self.network_s_list) - 1):
            if i % 6 == 0:
                ui = self.network_s_list[i](s_list[-1])
            elif i % 6 == 1:
                ui = self.network_s_list[i](s_list[-2])
            elif i % 6 == 2:
                ui, layer_attn = self.network_s_list[i](s_list[-3], s_list[-2], s_list[-1], attention_mask)
                attn_list.append(layer_attn)
            elif i % 6 == 3:
                ui = self.network_s_list[i](s_list[-1], s_list[-4])
            elif i % 6 == 5:
                ui = self.network_s_list[i](s_list[-1], s_list[-2])
            else:
                ui = self.network_s_list[i](s_list[-1])

            if i%6 in [0,1]:
                si = (ui >= fac*self.vth).float()
                ui = ui - fac*self.vth * si
            else:
                si = (ui >= self.vth).float()
                ui = ui - self.vth * si
            
            ui = ui * self.leaky
            
            u_list.append(ui)
            s_list.append(si)

        af = s_list[0]
        al = s_list[-self.fb_num]

        # a_per_layer 초기화
        a_per_layer = []
        avg_attn = []
        for i in range(len(s_list)):
            a_per_layer.append(s_list[i])
        for i in range(len(attn_list)):
            avg_attn.append(attn_list[i])

        if output_type == 'all_rate':
            r_list = []
            for s in s_list:
                r_list.append(s)

        # 나머지 타임스텝 처리
        for t in range(time_step - 1):
            # GPU 캐시 정리는 메모리 부족 시에만 수행
            if t % 10 == 0:  # 주기적으로만 수행
                torch.cuda.empty_cache()
            
            if input_type == 'constant':
                u_list[0] = u_list[0] + x1
            
            s_list[0] = (u_list[0] >= self.vth).float()
            u_list[0] = u_list[0] - self.vth * s_list[0]
            u_list[0] = u_list[0] * self.leaky
            
            # 첫 번째 레이어 로깅
            #self._log_features(0, t+1, u_list[0], s_list[0], None, 0.0, train_loss)

            for i in range(len(self.network_s_list) - 1):
                if i % 6 == 0:  # Attention 레이어마다 예측 수행
                    #predicted_rate = self.predict_spike_update(i+1, u_list[i+1], s_list[i])
                    
                    #u_list[i + 1] = u_list[i + 1] + self.network_s_list[i](s_list[i]) + torch.tensor(predicted_rate).to(u_list[i + 1].device)
                    u_list[i + 1] = u_list[i + 1] + (self.network_s_list[i](s_list[i]) * 0.5)
                    #self._log_features(i+1, t+1, u_list[i + 1], s_list[i + 1], s_list[i], self.network_s_list[i](s_list[i]), train_loss)
                elif i % 6 == 1:
                    u_list[i + 1] = u_list[i + 1] + self.network_s_list[i](s_list[i-1])
                elif i % 6 == 2:
                    u_val, layer_attn = self.network_s_list[i](s_list[i-2], s_list[i-1], s_list[i], attention_mask)
                    u_list[i + 1] = u_list[i + 1] + u_val
                    attn_list[int(i / 6)] = layer_attn
                elif i % 6 == 3:
                    u_list[i + 1] = u_list[i + 1] + self.network_s_list[i](s_list[i], s_list[i-3])
                elif i % 6 == 5:
                    u_list[i + 1] = u_list[i + 1] + self.network_s_list[i](s_list[i], s_list[i-1])
                else:
                    u_list[i + 1] = u_list[i + 1] + self.network_s_list[i](s_list[i])

                if i%6 in [0,1]:
                    s_list[i + 1] = (u_list[i + 1] >= fac*self.vth).float()
                    u_list[i + 1] = u_list[i + 1] - fac * self.vth * s_list[i + 1]
                else:
                    s_list[i + 1] = (u_list[i + 1] >= self.vth).float()
                    u_list[i + 1] = u_list[i + 1] - self.vth * s_list[i + 1]
                
                u_list[i + 1] = u_list[i + 1] * self.leaky
                
                
                
                # 각 레이어 로깅
                #self._log_features(i+1, t+1, u_list[i + 1], s_list[i + 1], s_list[i], 0.0, train_loss)

            af = af * self.leaky + s_list[0]
            al = al * self.leaky + s_list[-self.fb_num]

            #self.predicted_update_rate = predicted_rate
            # 벡터화된 누적 연산
            if len(s_list) > 0:
                # 모든 레이어의 스파이크를 한 번에 누적
                a_per_layer = [a + s for a, s in zip(a_per_layer, s_list)]
                
            # 어텐션 값도 한 번에 누적
            if len(attn_list) > 0:
                avg_attn = [a + attn for a, attn in zip(avg_attn, attn_list)]

            if output_type == 'all_rate':
                for i in range(len(r_list)):
                    r_list[i] = r_list[i] + s_list[i]

        weighted = ((1. - self.leaky ** time_step) / (1. - self.leaky))
        
        if output_type == 'normal':
            return af / weighted, al / weighted
        elif output_type == 'all_layers':
            # 벡터화된 평균 계산
            scale = 1.0 / time_step
            a_per_layer = [a * scale for a in a_per_layer]
            mean_spikes = torch.mean(torch.stack([torch.mean(layer) for layer in a_per_layer]))
            return a_per_layer, avg_attn, mean_spikes
        elif output_type == 'all_rate':
            for i in range(len(r_list)):
                r_list[i] *= 1.0 / time_step
            return r_list
        elif output_type == 'first':
            return af / weighted
        else:
            return al / weighted

    def _init_logging(self):
        self.prev_u = None
        self.cumulative_spikes = {}
        self.csv_path = os.path.join(self.log_dir, 'spike_features_qqp.csv')
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w') as f:
                f.write('prev_spike_rate,mean_voltage,current_spike,'
                       'voltage_threshold_ratio,cumulative_spikes,'
                       'predicted_update_rate,train_loss\n')
    
    def _log_features(self, layer_idx, step, u, s, prev_s, update_rate, train_loss=None):
        features = self.prepare_features(layer_idx, u, s)[0]
        
        update_rate = update_rate if type(update_rate) == float else torch.mean(update_rate)
        spike_change = torch.mean(torch.abs(u-s))
        # CSV에 기록
        with open(self.csv_path, 'a') as f:
            f.write(f"{features[0]:.6f},{features[1]:.6f},{update_rate:.6f},"
                   f"{features[2]:.6f},{features[3]:.6f},"
                   f"{spike_change:.6f},{train_loss if train_loss is not None else 0.0:.6f}\n")


class SNNFC(nn.Module):

    def __init__(self, d_in, d_out, bias=False, need_resize=False, sizes=None, dropout=0.0):
        super(SNNFC, self).__init__()
        self.fc = nn.Linear(d_in, d_out, bias=bias)
        self.need_resize = need_resize
        self.sizes=sizes
        self.drop = nn.Dropout(dropout)

        self._initialize_weights()

    def forward(self, x):
        if self.need_resize:
            if self.sizes == None:
                sizes = x.size()
                B = sizes[0]
                x = torch.reshape(self.fc(x.reshape(B, -1)), sizes)
            else:
                B = x.size(0)
                self.sizes[0] = B
                x = torch.reshape(self.fc(x.reshape(B, -1)), self.sizes)
        else:
            x = self.fc(x)
        return self.drop(x)

    def forward_linear(self, x):
        if self.need_resize:
            if self.sizes == None:
                sizes = x.size()
                B = sizes[0]
                x = torch.reshape(self.fc(x.reshape(B, -1)), sizes)
            else:
                B = x.size(0)
                self.sizes[0] = B
                x = torch.reshape(self.fc(x.reshape(B, -1)), self.sizes)
        else:
            x = self.fc(x)
        return x

    def _wnorm(self, norm_range=1.):
        self.fc, self.fc_fn = weight_spectral_norm(self.fc, names=['weight'], dim=0, norm_range=norm_range)

    def _reset(self, x):
        if 'fc_fn' in self.__dict__:
            self.fc_fn.reset(self.fc)
        self.drop.reset_mask(x)

    def _initialize_weights(self):
        m = self.fc
        m.weight.data.uniform_(-1, 1)
        for i in range(m.weight.size(0)):
            m.weight.data[i] /= torch.norm(m.weight.data[i])
        if m.bias is not None:
            m.bias.data.zero_()

    def copy(self, target):
        self.fc.weight.data = target.fc.weight.data.clone()
        if self.fc.bias is not None:
            self.fc.bias.data = target.fc.bias.data.clone()
        self.need_resize = target.need_resize
        self.sizes = target.sizes 
 
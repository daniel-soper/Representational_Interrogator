# ========================================================================
# Implementation and Experimental Evaluation of the Representational 
# Interrogator (RI) and Coordinate Dependency Index (CDI).
#
# This code was written in support of the follwing paper:
# 
# Soper, D.S. (2026). "Metrics and a Single-Pass Framework for Disentangling 
# Feature-Level Privacy Risk in Machine Learning Models", Machine Learning &
# Knowledge Extraction (MAKE),
# ========================================================================

import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import cohen_kappa_score
from scipy.stats import spearmanr, kendalltau, entropy
import lightgbm as lgb
from tqdm import tqdm
import os

# RI-CDI Experiments

# ==========================================
# 1. CONFIGURATION & HYPERPARAMETERS
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ATTACK_FOCAL_GAMMA = 5.0
ATTACK_FOCAL_ALPHA = None  # No scaling (full magnitude)
AUDITOR_FOCAL_GAMMA = 2.0  
AUDITOR_FOCAL_ALPHA = 0.25
BATCH_SIZE = 64
CGAN_BETA_1 = 0.5 
CGAN_LAMBDA = 100.0        # High reconstruction penalty
LEARNING_RATE = 0.001
MAX_EPOCHS = 300
NUM_TRIALS = 30
PATIENCE = 10
START_TRIAL = 1

ACTIVE_DATASETS = ['GERMAN', 'SHOPPERS', 'ADULT'] 

DATA_CONFIG = {
    'GERMAN': {
        'path': 'german_credit.csv',
        'target': 'credit_risk', 
        'categorical': ['status', 'credit_history', 'purpose', 'savings', 'employment', 'personal_status', 'other_debtors', 'property', 'installment_plans', 'housing', 'job', 'telephone', 'foreign_worker'],
        'continuous': ['duration', 'credit_amount', 'installment_rate', 'residence_since', 'age', 'existing_credits', 'people_liable']
    },
    'SHOPPERS': {
        'path': 'online_shoppers_intention.csv',
        'target': 'Revenue',
        'categorical': ['Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType', 'Weekend'],
        'continuous': [
            'Administrative', 'Administrative_Duration', 
            'Informational', 'Informational_Duration', 
            'ProductRelated', 'ProductRelated_Duration', 
            'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay'
        ]
    },
    'ADULT': {
        'path': 'adult.csv', 
        'target': 'income',
        'categorical': ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country'],
        'continuous': ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    }
}

# ==========================================
# 2. UTILITIES & LOSS FUNCTIONS
# ==========================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class FocalLoss(nn.Module):
    def __init__(self, gamma=AUDITOR_FOCAL_GAMMA, alpha=AUDITOR_FOCAL_ALPHA):
        super(FocalLoss, self).__init__()
        self.gamma, self.alpha = gamma, alpha
    def forward(self, inputs, targets):
        if inputs.dim() > 1 and inputs.shape[1] > 1:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        else:
            ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
            
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.alpha is not None:
            focal_loss = self.alpha * focal_loss
        return focal_loss.mean()

# ==========================================
# 3. DATA HANDLER
# ==========================================
class DataHandler:
    def __init__(self, dataset_name):
        self.config = DATA_CONFIG[dataset_name]
        self.df = pd.read_csv(self.config['path'])
        
        # Fix for boolean columns in SHOPPERS dataset
        if dataset_name == 'SHOPPERS':
            self.df['Weekend'] = self.df['Weekend'].astype(str)
            self.df['Revenue'] = self.df['Revenue'].astype(int)

        self.target_enc = LabelEncoder()
        self.df[self.config['target']] = self.target_enc.fit_transform(self.df[self.config['target']])
        
        self.cat_cols = self.config['categorical']
        self.cont_cols = self.config['continuous']
        
        self.ct = ColumnTransformer([
            ('num', StandardScaler(), self.cont_cols),
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), self.cat_cols)
        ])
        
        self.X_processed = self.ct.fit_transform(self.df.drop(columns=[self.config['target']]))
        self.y = self.df[self.config['target']].values
        
        self.feature_metadata = []
        start_idx = 0
        
        for col in self.cont_cols:
            self.feature_metadata.append({
                'name': col, 'type': 'continuous', 'start': start_idx, 'dim': 1
            })
            start_idx += 1
            
        for i, col in enumerate(self.cat_cols):
            n_cats = len(self.ct.named_transformers_['cat'].categories_[i])
            self.feature_metadata.append({
                'name': col, 'type': 'categorical', 'start': start_idx, 'dim': n_cats
            })
            start_idx += n_cats

    def get_data(self):
        return self.X_processed, self.y

# ==========================================
# 4. VICTIM MODELS
# ==========================================
class VictimModel:
    def train(self, X, y): pass
    def predict_proba(self, X): pass

# Deep Neural Network Victim
class DNN_Victim(VictimModel):
    def __init__(self, input_dim):
        # Architecture: 3 Hidden Layers (128, 64, 32)
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 2) 
        ).to(DEVICE)
        self.criterion = FocalLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

    def train(self, X, y):
        self.model.train()
        dataset = TensorDataset(torch.FloatTensor(X).to(DEVICE), torch.LongTensor(y).to(DEVICE))
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        # Early Stopping Logic (Patience=10)
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(MAX_EPOCHS):
            epoch_loss = 0.0
            for bx, by in loader:
                self.optimizer.zero_grad()
                out = self.model(bx)
                loss = self.criterion(out, by)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(loader)
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= PATIENCE:
                break

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.FloatTensor(X).to(DEVICE))
            return F.softmax(logits, dim=1).cpu().numpy()

# Gradient-Boosted Decision Tree Victim
class GBDT_Victim(VictimModel):
    def __init__(self, input_dim=None):
        self.model = None

    def train(self, X, y):
        train_data = lgb.Dataset(X, label=y)
        params = {'objective': 'binary', 'metric': 'binary_logloss', 'verbosity': -1, 'seed': 42}
        self.model = lgb.train(params, train_data, num_boost_round=100)

    def predict_proba(self, X):
        return np.stack([1-self.model.predict(X), self.model.predict(X)], axis=1)

# =================================
# 5. Representational Interrogator 
# =================================
# Architecture: Single MTL Model with Shared Stem + N Heads -> O(1) Efficiency
class MTL_RepresentationalInterrogator(nn.Module):
    def __init__(self, y_dim, feature_meta):
        super().__init__()
        self.feature_meta = feature_meta
        
        # Shared Stem: (32, 64, 128)
        self.stem = nn.Sequential(
            nn.Linear(y_dim, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU()
        )
        
        # Decoder Heads: One head per feature
        self.heads = nn.ModuleList()
        for m in feature_meta:
            self.heads.append(nn.Linear(128, m['dim']))
            
    def forward(self, y):
        z = self.stem(y)
        outputs = []
        for head in self.heads:
            outputs.append(head(z))
        return outputs

class RepresentationalInterrogatorRunner:
    def __init__(self, X, Y_victim, feature_meta):
        self.X = X
        self.Y = Y_victim
        self.meta = feature_meta
        self.model = MTL_RepresentationalInterrogator(Y_victim.shape[1], feature_meta).to(DEVICE)
        
    def calculate_uncertainties(self):
        # Algorithm 1, Step 1: Establish Uncertainty Baselines (Ui)
        uncertainties = []
        for m in self.meta:
            data = self.X[:, m['start']:m['start']+m['dim']]
            if m['type'] == 'continuous':
                # Variance for continuous
                u = np.var(data)
            else:
                # Shannon Entropy for categorical (one-hot encoded)
                # Sum columns to get counts -> prob dist
                counts = np.sum(data, axis=0)
                probs = counts / (np.sum(counts) + 1e-10)
                u = entropy(probs)
            uncertainties.append(u)
        return torch.tensor(uncertainties, dtype=torch.float32).to(DEVICE)

    def execute_audit(self):
        start_time = time.time()
        
        # Step 4: Uncertainty-Normalized Weighting
        uncertainties = self.calculate_uncertainties()
        weights = 1.0 / (uncertainties + 1e-8)
        
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        Yt = torch.FloatTensor(self.Y).to(DEVICE)
        Xt = torch.FloatTensor(self.X).to(DEVICE)
        dataset = TensorDataset(Yt, Xt)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        best_loss = float('inf')
        patience_counter = 0
        
        # Single Training Pass for All Features
        for _ in range(MAX_EPOCHS):
            epoch_loss = 0.0
            for by, bx in loader:
                optimizer.zero_grad()
                outputs = self.model(by)
                
                total_loss = 0.0
                # Calculate loss for each head
                for i, (out, m, w) in enumerate(zip(outputs, self.meta, weights)):
                    # Slice ground truth for this feature
                    target = bx[:, m['start']:m['start']+m['dim']]
                    
                    if m['type'] == 'continuous':
                        l = nn.MSELoss()(out, target)
                    else:
                        l = FocalLoss(gamma=AUDITOR_FOCAL_GAMMA, alpha=AUDITOR_FOCAL_ALPHA)(out, torch.argmax(target, 1))
                    
                    # Weighted summation
                    total_loss += w * l
                
                total_loss.backward()
                optimizer.step()
                epoch_loss += total_loss.item()
            
            # Early Stopping Check
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= PATIENCE:
                break
                
        audit_time = time.time() - start_time
        
        # Step 5: Quantify CDI
        self.model.eval()
        cdi_vector = []
        uncertainty_vector = []
        
        with torch.no_grad():
            outputs = self.model(Yt)
            for i, (out, m, u) in enumerate(zip(outputs, self.meta, uncertainties)):
                target = Xt[:, m['start']:m['start']+m['dim']]
                
                if m['type'] == 'continuous':
                    mse = nn.MSELoss()(out, target).item()
                    # CDI = exp(-MSE/Var)
                    # Using the strict definition CDI = exp(-L/U)
                    cdi = np.exp(-mse / (u.item() + 1e-8))
                else:
                    probs = F.softmax(out, dim=1)
                    nll = F.nll_loss(torch.log(probs + 1e-8), torch.argmax(target, 1)).item()
                    cdi = np.exp(-nll / (u.item() + 1e-8))
                
                cdi_vector.append(cdi)
                uncertainty_vector.append(u.item())
                
        return audit_time, cdi_vector, uncertainty_vector

# ============================================================
# 6. STATE-oF-THE-ART ATTACK MODELS (Neural Inversion & cGAN)
# ============================================================
class NeuralInversion_Attack:
    def execute(self, Y_victim, target_feature, feature_type, U, pbar=None):
        start_time = time.time()
        y_dim, x_dim = Y_victim.shape[1], target_feature.shape[1]
        
        model = nn.Sequential(
            nn.Linear(y_dim, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, x_dim)
        ).to(DEVICE)
        
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        Yt = torch.FloatTensor(Y_victim).to(DEVICE)
        Xt = torch.FloatTensor(target_feature).to(DEVICE)
        dataset = TensorDataset(Yt, Xt)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(MAX_EPOCHS):
            epoch_loss = 0.0
            for by, bx in loader:
                optimizer.zero_grad()
                out = model(by)
                
                if feature_type == 'continuous':
                    loss = nn.MSELoss()(out, bx)
                else:
                    # Use Focal Loss
                    loss = FocalLoss(gamma=ATTACK_FOCAL_GAMMA, alpha=ATTACK_FOCAL_ALPHA)(out, torch.argmax(bx, 1))
                    
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if pbar: pbar.update(1) # Update progress bar
            
            avg_loss = epoch_loss / len(loader)
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= PATIENCE:
                if pbar: 
                    remaining = (MAX_EPOCHS - epoch - 1)
                    if remaining > 0: pbar.update(remaining)
                break
                
        model.eval()
        with torch.no_grad():
            pred = model(Yt)
            if feature_type == 'continuous':
                mse = nn.MSELoss()(pred, Xt).item()
                risk = 1.0 - (mse / (torch.var(Xt).item() + 1e-8))
            else:
                pred_classes = torch.argmax(pred, 1)
                true_classes = torch.argmax(Xt, 1)
                risk = cohen_kappa_score(true_classes.cpu().numpy(), pred_classes.cpu().numpy())
                
        return time.time() - start_time, risk

class cGAN_Attack:
    def execute(self, Y_victim, target_feature, feature_type, U, pbar=None):
        start_time = time.time()
        y_dim, x_dim = Y_victim.shape[1], target_feature.shape[1]
        
        # Generator
        G = nn.Sequential(
            nn.Linear(32 + y_dim, 64), 
            nn.ReLU(), 
            nn.Linear(64, 128), 
            nn.ReLU(), 
            nn.Linear(128, x_dim)
        ).to(DEVICE)
        
        # Discriminator
        D = nn.Sequential(
            nn.Linear(x_dim + y_dim, 128), 
            nn.LeakyReLU(0.2), 
            nn.Linear(128, 64), 
            nn.LeakyReLU(0.2), 
            nn.Linear(64, 1), 
            nn.Sigmoid()
        ).to(DEVICE)
        
        opt_G = optim.Adam(G.parameters(), lr=LEARNING_RATE, betas=(CGAN_BETA_1, 0.999))
        opt_D = optim.Adam(D.parameters(), lr=LEARNING_RATE, betas=(CGAN_BETA_1, 0.999))
        
        adv_loss = nn.BCELoss()
        Yt = torch.FloatTensor(Y_victim).to(DEVICE)
        Xt = torch.FloatTensor(target_feature).to(DEVICE)
        
        loader = DataLoader(TensorDataset(Yt, Xt), batch_size=BATCH_SIZE, shuffle=True)
        
        for _ in range(MAX_EPOCHS):
            for by, bx in loader:
                b_sz = by.shape[0]
                valid = torch.ones(b_sz, 1).to(DEVICE)
                fake = torch.zeros(b_sz, 1).to(DEVICE)
                
                # --- Train Generator ---
                opt_G.zero_grad()
                z = torch.randn(b_sz, 32).to(DEVICE)
                gen_logits = G(torch.cat((z, by), dim=1)) # Capture Logits
                
                if feature_type == 'categorical':
                    # 1. Reconstruction Loss: Use Aggressive Attack Settings
                    target_indices = torch.argmax(bx, 1)
                    g_loss_recon = FocalLoss(gamma=ATTACK_FOCAL_GAMMA, alpha=ATTACK_FOCAL_ALPHA)(gen_logits, target_indices)
                    
                    # 2. Adversarial Loss: Use Softmaxed Probs
                    gen_out = F.softmax(gen_logits, dim=1)
                    g_loss_adv = adv_loss(D(torch.cat((gen_out, by), dim=1)), valid)
                
                else: # continuous
                    gen_out = gen_logits
                    g_loss_recon = nn.MSELoss()(gen_out, bx)
                    g_loss_adv = adv_loss(D(torch.cat((gen_out, by), dim=1)), valid)
                
                # Use Constant CGAN_LAMBDA
                total_g_loss = g_loss_adv + (CGAN_LAMBDA * g_loss_recon)
                
                total_g_loss.backward()
                opt_G.step()
                
                # --- Train Discriminator ---
                opt_D.zero_grad()
                d_real = adv_loss(D(torch.cat((bx, by), dim=1)), valid)
                # Detach gen_out to avoid training G
                d_fake = adv_loss(D(torch.cat((gen_out.detach(), by), dim=1)), fake)
                d_loss = (d_real + d_fake) / 2
                d_loss.backward()
                opt_D.step()

            if pbar: pbar.update(1) # Update progress bar
            
        # --- Evaluation Loop ---
        G.eval()
        with torch.no_grad():
            N_SAMPLES = 100 
            
            if feature_type == 'continuous':
                recon_sum = torch.zeros_like(Xt)
                for _ in range(N_SAMPLES):
                    z = torch.randn(Yt.shape[0], 32).to(DEVICE)
                    recon_batch = G(torch.cat((z, Yt), dim=1))
                    recon_sum += recon_batch
                
                recon_avg = recon_sum / N_SAMPLES
                L_attack = nn.MSELoss()(recon_avg, Xt).item()
                risk = 1.0 - (L_attack / (torch.var(Xt).item() + 1e-8))
                
            else: 
                probs_sum = torch.zeros_like(Xt)
                for _ in range(N_SAMPLES):
                    z = torch.randn(Yt.shape[0], 32).to(DEVICE)
                    gen_logits = G(torch.cat((z, Yt), dim=1))
                    probs = F.softmax(gen_logits, dim=1)
                    probs_sum += probs
                
                probs_avg = probs_sum / N_SAMPLES
                pred_classes = torch.argmax(probs_avg, 1)
                true_classes = torch.argmax(Xt, 1)
                
                risk = cohen_kappa_score(true_classes.cpu().numpy(), pred_classes.cpu().numpy())
            
        return time.time() - start_time, risk

# ==========================================
# 7. MAIN EXPERIMENT LOOP
# ==========================================
def run_experiments():
    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"Running on device: {DEVICE}") 

    trial_summary_data = []

    for trial in range(START_TRIAL - 1, NUM_TRIALS):
        set_seed(42 + trial)
        print(f"\n{'='*20} TRIAL {trial+1} of {NUM_TRIALS} {'='*20}")
        
        for ds_name in ACTIVE_DATASETS:
            print(f"\n--- Processing Dataset: {ds_name} ---")
            handler = DataHandler(ds_name)
            X, y = handler.get_data()
            
            for m_idx, (m_type, V_cls) in enumerate([('DNN', DNN_Victim), ('GBDT', GBDT_Victim)]):
                print(f"\n  Victim: {m_type}")
                victim = V_cls(X.shape[1] if m_type == 'DNN' else None)
                victim.train(X, y)
                Y_victim = victim.predict_proba(X)
                
                # --- Auditor Phase ---
                print("    Running Representational Interrogator...")
                runner = RepresentationalInterrogatorRunner(X, Y_victim, handler.feature_metadata)
                t_ia, cdi_auditor, U = runner.execute_audit()
                
                # --- Attack Phase (Baseline O(N)) ---
                attacks = {
                    'Neural': {'obj': NeuralInversion_Attack(), 'risk': [], 'time': 0},
                    'cGAN':   {'obj': cGAN_Attack(), 'risk': [], 'time': 0}
                }
                
                feature_types = [m['type'] for m in handler.feature_metadata]
                
                for atk_name, data in attacks.items():
                    print(f"    Running {atk_name} Attack...")
                    with tqdm(total=len(handler.feature_metadata) * MAX_EPOCHS, leave=False) as pbar:
                        for i, m in enumerate(handler.feature_metadata):
                            pbar.set_description(f"      Current Feature: {m['name']}")
                            Xf = torch.FloatTensor(X).to(DEVICE)
                            Yf = torch.FloatTensor(Y_victim).to(DEVICE)
                            
                            target = X[:, m['start']:m['start']+m['dim']]
                            
                            t, risk = data['obj'].execute(Y_victim, target, m['type'], U[i], pbar=pbar)
                            data['time'] += t; data['risk'].append(risk)

                summary_row = {
                    'Dataset': ds_name, 
                    'Victim': m_type, 
                    'Trial': trial + 1, 
                    'Time_RI': t_ia, 
                    'Time_Neural': attacks['Neural']['time'], 
                    'Time_cGAN': attacks['cGAN']['time']
                }
                
                indices_cont = [i for i, t in enumerate(feature_types) if t == 'continuous']
                indices_cat = [i for i, t in enumerate(feature_types) if t == 'categorical']

                for atk_name in ['Neural', 'cGAN']:
                    print(f"\n      --- {atk_name} vs RI-CDI Statistics ---")
                    for dtype, indices in [("Continuous", indices_cont), ("Categorical", indices_cat)]:
                        if not indices: continue
                        
                        a_cdi = np.array(cdi_auditor)[indices]
                        atk_risk = np.array(attacks[atk_name]['risk'])[indices]
                        
                        # Adversarial clumping
                        if np.std(atk_risk) < 1e-9:
                            s_risk, k_risk = 0.0, 0.0 
                            print(f"        [{dtype}] (Risk) Warning: Constant attack output. Setting corr to 0.")
                        else:
                            s_risk = spearmanr(a_cdi, atk_risk)[0]
                            k_risk = kendalltau(a_cdi, atk_risk)[0]
                        
                        summary_row.update({
                            f'{atk_name}_{dtype}_Spearman_Risk': s_risk, 
                            f'{atk_name}_{dtype}_Kendall_Risk': k_risk
                        })
                        print(f"        [{dtype}] (Risk) Spearman: {s_risk:.4f}, Kendall: {k_risk:.4f}")

                trial_summary_data.append(summary_row)

                write_mode = 'w' if (trial == 0 and m_idx == 0) else 'a'
                write_header = True if (trial == 0 and m_idx == 0) else False
                
                df_results = pd.DataFrame({
                    'Trial': trial+1, 
                    'Victim': m_type, 
                    'Feature': [m['name'] for m in handler.feature_metadata], 
                    'Type': feature_types, 
                    'RI_CDI': cdi_auditor, 
                    'Neural_Risk': attacks['Neural']['risk'], 
                    'cGAN_Risk': attacks['cGAN']['risk']
                })
                df_results.to_csv(f'results_{ds_name}.csv', mode=write_mode, header=write_header, index=False)

        summary_write_mode = 'w' if trial == 0 else 'a'
        summary_write_header = True if trial == 0 else False
        df_summary = pd.DataFrame(trial_summary_data)
        df_summary.to_csv('experiment_summary.csv', mode=summary_write_mode, header=summary_write_header, index=False)
        trial_summary_data = [] 

        print(f"\nResults for Trial {trial+1} saved!")

    print(f"\nExperiments complete! All results were saved successfully!\n")

if __name__ == "__main__":
    run_experiments()
#!/bin/bash
# Football AI System - Phase 3: ML Models & Training Setup
# RTX 3090 Optimized Machine Learning Infrastructure

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging functions
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

# Check if running as footballai user
check_user() {
    if [ "$USER" != "footballai" ]; then
        error "Bitte als footballai Benutzer ausführen (use: su - footballai)"
        exit 1
    fi
}

# Activate conda environment
activate_env() {
    log "🐍 Aktiviere Python Environment..."
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate football-ai
    log "✅ Environment aktiviert"
}

# Create model base classes
create_model_bases() {
    log "🧠 Erstelle Model Base Classes..."
    
    cat > ~/football-ai-system/backend/models/base.py << 'EOF'
"""
Base classes for all ML models - RTX 3090 Optimized
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
import joblib
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """Base class for all prediction models"""
    
    def __init__(self, model_name: str, model_type: str, **kwargs):
        self.model_name = model_name
        self.model_type = model_type
        self.is_trained = False
        self.model = None
        self.feature_names = None
        self.target_names = None
        self.training_params = {}
        self.model_metadata = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # RTX 3090 optimizations
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.gpu_memory_fraction = kwargs.get('gpu_memory_fraction', 0.85)
            self.max_batch_size = kwargs.get('max_batch_size', 128)
            
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray, 
              validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
              **kwargs) -> Dict[str, float]:
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Make probability predictions"""
        pass
    
    def save_model(self, filepath: str) -> None:
        """Save model to disk"""
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            model_data = {
                'model_name': self.model_name,
                'model_type': self.model_type,
                'is_trained': self.is_trained,
                'feature_names': self.feature_names,
                'target_names': self.target_names,
                'training_params': self.training_params,
                'model_metadata': self.model_metadata,
                'created_at': datetime.now().isoformat()
            }
            
            # Save model-specific data
            if hasattr(self, 'model') and self.model is not None:
                if isinstance(self.model, nn.Module):  # PyTorch model
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'model_data': model_data
                    }, filepath)
                else:  # Scikit-learn or other models
                    joblib.dump({
                        'model': self.model,
                        'model_data': model_data
                    }, filepath)
            else:
                # Save metadata only
                with open(filepath, 'wb') as f:
                    joblib.dump({'model_data': model_data}, f)
            
            logger.info(f"✅ Model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save model: {e}")
            raise
    
    def load_model(self, filepath: str) -> None:
        """Load model from disk"""
        try:
            if not Path(filepath).exists():
                raise FileNotFoundError(f"Model file not found: {filepath}")
            
            # Load model data
            model_data = joblib.load(filepath)
            
            if 'model_data' in model_data:
                # Restore model metadata
                for key, value in model_data['model_data'].items():
                    if hasattr(self, key):
                        setattr(self, key, value)
                
                # Load actual model if present
                if 'model' in model_data:
                    self.model = model_data['model']
                
                logger.info(f"✅ Model loaded from {filepath}")
            else:
                logger.warning(f"⚠️ No model data found in {filepath}")
                
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'feature_names': self.feature_names,
            'target_names': self.target_names,
            'training_params': self.training_params,
            'device': str(self.device),
            'gpu_memory_fraction': getattr(self, 'gpu_memory_fraction', None),
            'max_batch_size': getattr(self, 'max_batch_size', None)
        }

class PyTorchModel(BaseModel):
    """Base class for PyTorch models"""
    
    def __init__(self, model_name: str, model_type: str, **kwargs):
        super().__init__(model_name, model_type, **kwargs)
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        
    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to device"""
        return tensor.to(self.device)
    
    def train_epoch(self, dataloader: torch.utils.data.DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = self.to_device(data), self.to_device(target)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Clear cache for RTX 3090
            if batch_idx % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return total_loss / len(dataloader)
    
    def validate(self, dataloader: torch.utils.data.DataLoader) -> float:
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = self.to_device(data), self.to_device(target)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
        
        return total_loss / len(dataloader)

class SklearnModel(BaseModel):
    """Base class for scikit-learn models"""
    
    def __init__(self, model_name: str, model_type: str, **kwargs):
        super().__init__(model_name, model_type, **kwargs)
        
    def train(self, X: np.ndarray, y: np.ndarray, 
              validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
              **kwargs) -> Dict[str, float]:
        """Train scikit-learn model"""
        try:
            self.model.fit(X, y)
            self.is_trained = True
            
            # Calculate training metrics
            train_score = self.model.score(X, y)
            metrics = {'train_score': train_score}
            
            # Calculate validation metrics if provided
            if validation_data:
                X_val, y_val = validation_data
                val_score = self.model.score(X_val, y_val)
                metrics['val_score'] = val_score
            
            logger.info(f"✅ {self.model_name} training completed")
            return metrics
            
        except Exception as e:
            logger.error(f"❌ {self.model_name} training failed: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError(f"Model {self.model_name} is not trained")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Make probability predictions"""
        if not self.is_trained:
            raise ValueError(f"Model {self.model_name} is not trained")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # Fallback for models without predict_proba
            predictions = self.predict(X)
            # Convert to one-hot encoding
            proba = np.zeros((len(predictions), len(np.unique(predictions))))
            proba[np.arange(len(predictions)), predictions] = 1.0
            return proba
EOF

    log "✅ Model Base Classes erstellt"
}

# Create neural network models
create_neural_networks() {
    log "🧠 Erstelle Neural Network Models..."
    
    # Correct Score Prediction Model
    cat > ~/football-ai-system/backend/models/neural_nets/correct_score_model.py << 'EOF'
"""
Correct Score Prediction Neural Network
Optimized for RTX 3090 with mixed precision training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np
from pathlib import Path

from ..base import PyTorchModel

class CorrectScoreNet(nn.Module):
    """Neural network for correct score prediction"""
    
    def __init__(self, input_dim: int, hidden_dims: list = [512, 256, 128], dropout_rate: float = 0.3):
        super().__init__()
        
        # Feature extraction layers
        self.feature_layers = nn.ModuleList()
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            self.feature_layers.append(nn.Linear(prev_dim, hidden_dim))
            self.feature_layers.append(nn.BatchNorm1d(hidden_dim))
            self.feature_layers.append(nn.ReLU())
            self.feature_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Score prediction branches
        # Home goals (0-10)
        self.home_goals_branch = nn.Sequential(
            nn.Linear(prev_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 11)  # 0-10 goals
        )
        
        # Away goals (0-10)
        self.away_goals_branch = nn.Sequential(
            nn.Linear(prev_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 11)  # 0-10 goals
        )
        
        # Match outcome (Win, Draw, Loss)
        self.outcome_branch = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # Home Win, Draw, Away Win
        )
        
        # Total goals (0-10)
        self.total_goals_branch = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 11)  # 0-10 total goals
        )
    
    def forward(self, x):
        # Feature extraction
        features = x
        for layer in self.feature_layers:
            features = layer(features)
        
        # Multiple outputs
        home_goals_logits = self.home_goals_branch(features)
        away_goals_logits = self.away_goals_branch(features)
        outcome_logits = self.outcome_branch(features)
        total_goals_logits = self.total_goals_branch(features)
        
        return {
            'home_goals': home_goals_logits,
            'away_goals': away_goals_logits,
            'outcome': outcome_logits,
            'total_goals': total_goals_logits
        }

class CorrectScoreModel(PyTorchModel):
    """Correct Score Prediction Model"""
    
    def __init__(self, model_name: str = "correct_score_net", **kwargs):
        super().__init__(model_name, "correct_score", **kwargs)
        self.input_dim = kwargs.get('input_dim', 100)
        self.hidden_dims = kwargs.get('hidden_dims', [512, 256, 128])
        self.dropout_rate = kwargs.get('dropout_rate', 0.3)
        self.learning_rate = kwargs.get('learning_rate', 0.001)
        
        # Initialize model
        self.model = CorrectScoreNet(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            dropout_rate=self.dropout_rate
        )
        
        # Move to device
        self.model = self.model.to(self.device)
        
        # Initialize optimizer and criterion
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate,
            weight_decay=1e-5
        )
        
        # Multiple loss functions for multi-task learning
        self.criteria = {
            'home_goals': nn.CrossEntropyLoss(),
            'away_goals': nn.CrossEntropyLoss(),
            'outcome': nn.CrossEntropyLoss(),
            'total_goals': nn.CrossEntropyLoss()
        }
        
        # Loss weights for balanced learning
        self.loss_weights = {
            'home_goals': 1.0,
            'away_goals': 1.0,
            'outcome': 2.0,  # Higher weight for outcome
            'total_goals': 0.8
        }
    
    def forward_loss(self, outputs: Dict[str, torch.Tensor], 
                    targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculate weighted loss for multi-task learning"""
        total_loss = 0.0
        
        for task, criterion in self.criteria.items():
            if task in outputs and task in targets:
                loss = criterion(outputs[task], targets[task])
                weighted_loss = self.loss_weights[task] * loss
                total_loss += weighted_loss
        
        return total_loss
    
    def train(self, X: np.ndarray, y: Dict[str, np.ndarray], 
              validation_data: Optional[Tuple] = None,
              epochs: int = 100, batch_size: int = 64) -> Dict[str, float]:
        """Train the correct score model"""
        
        try:
            # Convert to tensors
            X_tensor = torch.FloatTensor(X).to(self.device)
            y_tensors = {
                k: torch.LongTensor(v).to(self.device)
                for k, v in y.items()
            }
            
            # Create dataset and dataloader
            dataset = torch.utils.data.TensorDataset(X_tensor, *y_tensors.values())
            dataloader = torch.utils.data.DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=True,
                num_workers=2,
                pin_memory=True if torch.cuda.is_available() else False
            )
            
            # Training loop
            self.model.train()
            train_losses = []
            
            for epoch in range(epochs):
                epoch_loss = 0.0
                batch_count = 0
                
                for batch in dataloader:
                    batch_X = batch[0]
                    batch_targets = {
                        'home_goals': batch[1],
                        'away_goals': batch[2],
                        'outcome': batch[3],
                        'total_goals': batch[4]
                    }
                    
                    # Forward pass
                    self.optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = self.forward_loss(outputs, batch_targets)
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    self.optimizer.step()
                    
                    epoch_loss += loss.item()
                    batch_count += 1
                    
                    # Clear cache for RTX 3090
                    if batch_count % 10 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                avg_epoch_loss = epoch_loss / batch_count
                train_losses.append(avg_epoch_loss)
                
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}/{epochs}, Loss: {avg_epoch_loss:.4f}")
            
            self.is_trained = True
            
            return {
                'final_loss': train_losses[-1],
                'min_loss': min(train_losses),
                'epochs_trained': epochs
            }
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model is not trained")
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            
            # Batch processing for memory efficiency
            predictions = {
                'home_goals': [],
                'away_goals': [],
                'outcome': [],
                'total_goals': []
            }
            
            batch_size = min(len(X), self.max_batch_size)
            
            for i in range(0, len(X), batch_size):
                batch_X = X_tensor[i:i+batch_size]
                batch_outputs = self.model(batch_X)
                
                # Convert to predictions
                for task in predictions.keys():
                    task_pred = torch.argmax(batch_outputs[task], dim=1).cpu().numpy()
                    predictions[task].extend(task_pred)
                
                # Clear cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            return {k: np.array(v) for k, v in predictions.items()}
    
    def predict_proba(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Make probability predictions"""
        if not self.is_trained:
            raise ValueError("Model is not trained")
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            
            probabilities = {}
            batch_size = min(len(X), self.max_batch_size)
            
            for i in range(0, len(X), batch_size):
                batch_X = X_tensor[i:i+batch_size]
                batch_outputs = self.model(batch_X)
                
                # Apply softmax to get probabilities
                for task, output in batch_outputs.items():
                    task_probs = torch.softmax(output, dim=1).cpu().numpy()
                    
                    if task not in probabilities:
                        probabilities[task] = []
                    probabilities[task].extend(task_probs)
                
                # Clear cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            return {k: np.array(v) for k, v in probabilities.items()}

# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    X_sample = np.random.randn(1000, 100).astype(np.float32)
    y_sample = {
        'home_goals': np.random.randint(0, 6, 1000),
        'away_goals': np.random.randint(0, 6, 1000),
        'outcome': np.random.randint(0, 3, 1000),
        'total_goals': np.random.randint(0, 7, 1000)
    }
    
    # Initialize and train model
    model = CorrectScoreModel(input_dim=100)
    
    print("Training Correct Score Model...")
    results = model.train(X_sample, y_sample, epochs=5)
    print(f"Training completed: {results}")
    
    # Make predictions
    test_X = np.random.randn(10, 100).astype(np.float32)
    predictions = model.predict(test_X)
    print(f"Predictions: {predictions}")
    
    probabilities = model.predict_proba(test_X)
    print(f"Probabilities shape: {probabilities['home_goals'].shape}")
EOF

    # Over/Under Model
    cat > ~/football-ai-system/backend/models/neural_nets/over_under_model.py << 'EOF'
"""
Over/Under Goals Prediction Neural Network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np

from ..base import PyTorchModel

class OverUnderNet(nn.Module):
    """Neural network for over/under goals prediction"""
    
    def __init__(self, input_dim: int, hidden_dims: list = [256, 128, 64], dropout_rate: float = 0.3):
        super().__init__()
        
        # Shared feature extraction
        self.feature_layers = nn.ModuleList()
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            self.feature_layers.append(nn.Linear(prev_dim, hidden_dim))
            self.feature_layers.append(nn.BatchNorm1d(hidden_dim))
            self.feature_layers.append(nn.ReLU())
            self.feature_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Over/Under predictions for different thresholds
        self.thresholds = [0.5, 1.5, 2.5, 3.5, 4.5]
        self.over_under_branches = nn.ModuleDict()
        
        for threshold in self.thresholds:
            self.over_under_branches[str(threshold)] = nn.Sequential(
                nn.Linear(prev_dim, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 2)  # Over/Under
            )
        
        # Total goals regression
        self.total_goals_regression = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        # Feature extraction
        features = x
        for layer in self.feature_layers:
            features = layer(features)
        
        # Multiple over/under predictions
        outputs = {}
        for threshold, branch in self.over_under_branches.items():
            outputs[f'over_under_{threshold}'] = branch(features)
        
        # Total goals regression
        outputs['total_goals'] = self.total_goals_regression(features)
        
        return outputs

class OverUnderModel(PyTorchModel):
    """Over/Under Goals Prediction Model"""
    
    def __init__(self, model_name: str = "over_under_net", **kwargs):
        super().__init__(model_name, "over_under", **kwargs)
        self.input_dim = kwargs.get('input_dim', 100)
        self.hidden_dims = kwargs.get('hidden_dims', [256, 128, 64])
        self.dropout_rate = kwargs.get('dropout_rate', 0.3)
        self.learning_rate = kwargs.get('learning_rate', 0.001)
        
        # Thresholds for over/under predictions
        self.thresholds = [0.5, 1.5, 2.5, 3.5, 4.5]
        
        # Initialize model
        self.model = OverUnderNet(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            dropout_rate=self.dropout_rate
        )
        
        # Move to device
        self.model = self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate,
            weight_decay=1e-5
        )
        
        # Loss functions
        self.criteria = {
            'classification': nn.CrossEntropyLoss(),
            'regression': nn.MSELoss()
        }
        
        # Loss weights
        self.loss_weights = {
            'classification': 1.0,
            'regression': 0.5
        }
    
    def prepare_targets(self, y: np.ndarray) -> Dict[str, torch.Tensor]:
        """Prepare targets for different thresholds"""
        targets = {}
        
        # Total goals for regression
        targets['total_goals'] = torch.FloatTensor(y).to(self.device)
        
        # Over/Under for each threshold
        for threshold in self.thresholds:
            over_under = (y > threshold).astype(int)
            targets[f'over_under_{threshold}'] = torch.LongTensor(over_under).to(self.device)
        
        return targets
    
    def forward_loss(self, outputs: Dict[str, torch.Tensor], 
                    targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculate loss for over/under model"""
        total_loss = 0.0
        
        # Classification losses
        for threshold in self.thresholds:
            task_key = f'over_under_{threshold}'
            if task_key in outputs and task_key in targets:
                loss = self.criteria['classification'](outputs[task_key], targets[task_key])
                total_loss += self.loss_weights['classification'] * loss
        
        # Regression loss
        if 'total_goals' in outputs and 'total_goals' in targets:
            regression_loss = self.criteria['regression'](
                outputs['total_goals'].squeeze(), 
                targets['total_goals']
            )
            total_loss += self.loss_weights['regression'] * regression_loss
        
        return total_loss
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              validation_data: Optional[Tuple] = None,
              epochs: int = 100, batch_size: int = 64) -> Dict[str, float]:
        """Train the over/under model"""
        
        try:
            # Convert to tensors
            X_tensor = torch.FloatTensor(X).to(self.device)
            
            # Create dataset and dataloader
            dataset = torch.utils.data.TensorDataset(X_tensor)
            dataloader = torch.utils.data.DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=True,
                num_workers=2
            )
            
            # Prepare targets
            targets = self.prepare_targets(y)
            
            # Training loop
            self.model.train()
            train_losses = []
            
            for epoch in range(epochs):
                epoch_loss = 0.0
                batch_count = 0
                
                for batch in dataloader:
                    batch_X = batch[0]
                    batch_size_actual = batch_X.size(0)
                    
                    # Get corresponding targets
                    batch_targets = {}
                    start_idx = batch_count * batch_size
                    end_idx = start_idx + batch_size_actual
                    
                    for key, target_tensor in targets.items():
                        batch_targets[key] = target_tensor[start_idx:end_idx]
                    
                    # Forward pass
                    self.optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = self.forward_loss(outputs, batch_targets)
                    
                    # Backward pass
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    
                    epoch_loss += loss.item()
                    batch_count += 1
                    
                    # Clear cache
                    if batch_count % 10 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                avg_epoch_loss = epoch_loss / batch_count
                train_losses.append(avg_epoch_loss)
                
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}/{epochs}, Loss: {avg_epoch_loss:.4f}")
            
            self.is_trained = True
            
            return {
                'final_loss': train_losses[-1],
                'min_loss': min(train_losses),
                'epochs_trained': epochs
            }
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model is not trained")
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            
            predictions = {}
            batch_size = min(len(X), self.max_batch_size)
            
            for i in range(0, len(X), batch_size):
                batch_X = X_tensor[i:i+batch_size]
                batch_outputs = self.model(batch_X)
                
                # Process outputs
                for key, output in batch_outputs.items():
                    if 'over_under' in key:
                        # Classification
                        pred = torch.argmax(output, dim=1).cpu().numpy()
                        if key not in predictions:
                            predictions[key] = []
                        predictions[key].extend(pred)
                    elif key == 'total_goals':
                        # Regression
                        pred = output.squeeze().cpu().numpy()
                        if key not in predictions:
                            predictions[key] = []
                        predictions[key].extend(pred)
                
                # Clear cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            return {k: np.array(v) for k, v in predictions.items()}
    
    def predict_proba(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Make probability predictions"""
        if not self.is_trained:
            raise ValueError("Model is not trained")
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            
            probabilities = {}
            batch_size = min(len(X), self.max_batch_size)
            
            for i in range(0, len(X), batch_size):
                batch_X = X_tensor[i:i+batch_size]
                batch_outputs = self.model(batch_X)
                
                # Apply softmax to classification outputs
                for key, output in batch_outputs.items():
                    if 'over_under' in key:
                        task_probs = torch.softmax(output, dim=1).cpu().numpy()
                        if key not in probabilities:
                            probabilities[key] = []
                        probabilities[key].extend(task_probs)
                
                # Clear cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            return {k: np.array(v) for k, v in probabilities.items()}

# Example usage
if __name__ == "__main__":
    # Create sample data
    X_sample = np.random.randn(1000, 100).astype(np.float32)
    y_sample = np.random.randint(0, 6, 1000) + np.random.randint(0, 6, 1000)  # Total goals
    
    # Initialize and train model
    model = OverUnderModel(input_dim=100)
    
    print("Training Over/Under Model...")
    results = model.train(X_sample, y_sample, epochs=5)
    print(f"Training completed: {results}")
    
    # Make predictions
    test_X = np.random.randn(10, 100).astype(np.float32)
    predictions = model.predict(test_X)
    print(f"Predictions keys: {list(predictions.keys())}")
    
    probabilities = model.predict_proba(test_X)
    print(f"Probabilities keys: {list(probabilities.keys())}")
EOF

    log "✅ Neural Network Models erstellt"
}

# Create traditional ML models
create_traditional_models() {
    log "🌲 Erstelle Traditional ML Models..."
    
    # XGBoost Model
    cat > ~/football-ai-system/backend/models/traditional/xgboost_model.py << 'EOF'
"""
XGBoost Models for Football Predictions
"""

import xgboost as xgb
from typing import Dict, Tuple, Optional, Any
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

from ..base import SklearnModel

class XGBoostFootballModel(SklearnModel):
    """XGBoost model for football predictions"""
    
    def __init__(self, model_name: str, model_type: str, **kwargs):
        super().__init__(model_name, model_type, **kwargs)
        
        # XGBoost parameters optimized for RTX 3090
        self.params = {
            'n_estimators': kwargs.get('n_estimators', 1000),
            'max_depth': kwargs.get('max_depth', 6),
            'learning_rate': kwargs.get('learning_rate', 0.1),
            'subsample': kwargs.get('subsample', 0.8),
            'colsample_bytree': kwargs.get('colsample_bytree', 0.8),
            'reg_alpha': kwargs.get('reg_alpha', 0.1),
            'reg_lambda': kwargs.get('reg_lambda', 0.1),
            'random_state': kwargs.get('random_state', 42),
            'tree_method': 'gpu_hist' if torch.cuda.is_available() else 'hist',
            'gpu_id': 0 if torch.cuda.is_available() else -1,
            'predictor': 'gpu_predictor' if torch.cuda.is_available() else 'cpu_predictor'
        }
        
        # Initialize model based on model type
        if model_type == 'classification':
            self.model = xgb.XGBClassifier(**self.params)
        elif model_type == 'regression':
            self.model = xgb.XGBRegressor(**self.params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
              **kwargs) -> Dict[str, float]:
        """Train XGBoost model with GPU acceleration"""
        
        try:
            # Prepare validation set
            eval_set = None
            if validation_data:
                eval_set = [(validation_data[0], validation_data[1])]
            
            # Training parameters
            train_params = {
                'X': X,
                'y': y,
                'eval_set': eval_set,
                'verbose': kwargs.get('verbose', False),
                'early_stopping_rounds': kwargs.get('early_stopping_rounds', 50)
            }
            
            # Train model
            self.model.fit(**train_params)
            self.is_trained = True
            
            # Calculate metrics
            train_pred = self.model.predict(X)
            train_score = accuracy_score(y, train_pred) if self.model_type == 'classification' else \
                         np.mean(np.abs(y - train_pred))  # MAE for regression
            
            metrics = {
                'train_score': train_score,
                'best_iteration': self.model.best_iteration,
                'n_features': self.model.n_features_in_
            }
            
            # Validation metrics if provided
            if validation_data:
                X_val, y_val = validation_data
                val_pred = self.model.predict(X_val)
                val_score = accuracy_score(y_val, val_pred) if self.model_type == 'classification' else \
                           np.mean(np.abs(y_val - val_pred))
                metrics['val_score'] = val_score
                metrics['val_best_score'] = self.model.best_score
            
            logger.info(f"✅ {self.model_name} training completed")
            return metrics
            
        except Exception as e:
            logger.error(f"❌ {self.model_name} training failed: {e}")
            raise
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance"""
        if not self.is_trained:
            raise ValueError("Model is not trained")
        
        return self.model.feature_importances_
    
    def plot_feature_importance(self, feature_names: Optional[List[str]] = None) -> None:
        """Plot feature importance"""
        if not self.is_trained:
            raise ValueError("Model is not trained")
        
        import matplotlib.pyplot as plt
        
        importance = self.get_feature_importance()
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(importance))]
        
        # Create plot
        plt.figure(figsize=(12, 8))
        plt.barh(feature_names, importance)
        plt.xlabel('Feature Importance')
        plt.title(f'Feature Importance - {self.model_name}')
        plt.tight_layout()
        
        # Save plot
        plot_path = Path(f"models/feature_importance_{self.model_name}.png")
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path)
        plt.close()
        
        logger.info(f"Feature importance plot saved to {plot_path}")

class LightGBMFootballModel(SklearnModel):
    """LightGBM model for football predictions"""
    
    def __init__(self, model_name: str, model_type: str, **kwargs):
        super().__init__(model_name, model_type, **kwargs)
        
        # LightGBM parameters
        self.params = {
            'n_estimators': kwargs.get('n_estimators', 1000),
            'max_depth': kwargs.get('max_depth', -1),
            'learning_rate': kwargs.get('learning_rate', 0.1),
            'num_leaves': kwargs.get('num_leaves', 31),
            'subsample': kwargs.get('subsample', 0.8),
            'colsample_bytree': kwargs.get('colsample_bytree', 0.8),
            'reg_alpha': kwargs.get('reg_alpha', 0.1),
            'reg_lambda': kwargs.get('reg_lambda', 0.1),
            'random_state': kwargs.get('random_state', 42),
            'device': 'gpu' if torch.cuda.is_available() else 'cpu',
            'gpu_platform_id': 0 if torch.cuda.is_available() else -1,
            'gpu_device_id': 0 if torch.cuda.is_available() else -1
        }
        
        import lightgbm as lgb
        
        # Initialize model based on model type
        if model_type == 'classification':
            self.model = lgb.LGBMClassifier(**self.params)
        elif model_type == 'regression':
            self.model = lgb.LGBMRegressor(**self.params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

class CatBoostFootballModel(SklearnModel):
    """CatBoost model for football predictions"""
    
    def __init__(self, model_name: str, model_type: str, **kwargs):
        super().__init__(model_name, model_type, **kwargs)
        
        # CatBoost parameters
        self.params = {
            'iterations': kwargs.get('iterations', 1000),
            'depth': kwargs.get('depth', 6),
            'learning_rate': kwargs.get('learning_rate', 0.1),
            'l2_leaf_reg': kwargs.get('l2_leaf_reg', 3.0),
            'random_strength': kwargs.get('random_strength', 1),
            'bagging_temperature': kwargs.get('bagging_temperature', 1.0),
            'random_state': kwargs.get('random_state', 42),
            'task_type': 'GPU' if torch.cuda.is_available() else 'CPU',
            'devices': '0' if torch.cuda.is_available() else None
        }
        
        import catboost as cb
        
        # Initialize model based on model type
        if model_type == 'classification':
            self.model = cb.CatBoostClassifier(**self.params)
        elif model_type == 'regression':
            self.model = cb.CatBoostRegressor(**self.params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

# Model factory
class ModelFactory:
    """Factory for creating ML models"""
    
    @staticmethod
    def create_model(model_type: str, model_name: str, **kwargs) -> BaseModel:
        """Create a model instance"""
        
        model_mapping = {
            # Neural Networks
            'correct_score': 'CorrectScoreModel',
            'over_under': 'OverUnderModel',
            'btts': 'BothTeamsToScoreModel',
            
            # Traditional ML
            'xgboost': 'XGBoostFootballModel',
            'lightgbm': 'LightGBMFootballModel',
            'catboost': 'CatBoostFootballModel',
            'random_forest': 'RandomForestModel',
            'logistic_regression': 'LogisticRegressionModel'
        }
        
        if model_type not in model_mapping:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Import and create model
        if model_type in ['correct_score', 'over_under', 'btts']:
            from .neural_nets import globals()
            model_class = globals()[model_mapping[model_type]]
        else:
            from .traditional import globals()
            model_class = globals()[model_mapping[model_type]]
        
        return model_class(model_name, model_type, **kwargs)

# Example usage
if __name__ == "__main__":
    # Create sample data
    X_sample = np.random.randn(1000, 50)
    y_sample_classification = np.random.randint(0, 3, 1000)
    y_sample_regression = np.random.randn(1000)
    
    # Test XGBoost models
    xgb_classifier = XGBoostFootballModel("test_xgb_classifier", "classification")
    xgb_regressor = XGBoostFootballModel("test_xgb_regressor", "regression")
    
    print("Training XGBoost Classifier...")
    results = xgb_classifier.train(X_sample, y_sample_classification)
    print(f"Training results: {results}")
    
    print("Training XGBoost Regressor...")
    results = xgb_regressor.train(X_sample, y_sample_regression)
    print(f"Training results: {results}")
    
    # Make predictions
    test_X = np.random.randn(10, 50)
    predictions = xgb_classifier.predict(test_X)
    print(f"Classification predictions: {predictions}")
    
    predictions = xgb_regressor.predict(test_X)
    print(f"Regression predictions: {predictions}")
EOF

    log "✅ Traditional ML Models erstellt"
}

# Create ensemble models
create_ensemble_models() {
    log "🎯 Erstelle Ensemble Models..."
    
    cat > ~/football-ai-system/backend/models/ensemble/ensemble_models.py << 'EOF'
"""
Ensemble Models for Football Predictions
"""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from sklearn.ensemble import VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib

from ..base import BaseModel

class EnsembleModel(BaseModel):
    """Base class for ensemble models"""
    
    def __init__(self, model_name: str, model_type: str, **kwargs):
        super().__init__(model_name, model_type, **kwargs)
        self.base_models = []
        self.meta_model = None
        self.ensemble_method = kwargs.get('ensemble_method', 'voting')
        self.voting_type = kwargs.get('voting_type', 'soft')
    
    def add_base_model(self, model: BaseModel) -> None:
        """Add a base model to the ensemble"""
        self.base_models.append(model)
    
    def remove_base_model(self, model_name: str) -> None:
        """Remove a base model from the ensemble"""
        self.base_models = [m for m in self.base_models if m.model_name != model_name]
    
    def get_base_models(self) -> List[BaseModel]:
        """Get all base models"""
        return self.base_models
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
              **kwargs) -> Dict[str, float]:
        """Train ensemble model"""
        
        if not self.base_models:
            raise ValueError("No base models added to ensemble")
        
        # Train all base models
        base_model_scores = {}
        
        for i, base_model in enumerate(self.base_models):
            logger.info(f"Training base model {i+1}/{len(self.base_models)}: {base_model.model_name}")
            
            # Train base model
            if validation_data:
                score = base_model.train(X, y, validation_data)
            else:
                score = base_model.train(X, y)
            
            base_model_scores[base_model.model_name] = score
        
        # Create ensemble predictions
        ensemble_predictions = self._get_ensemble_predictions(X)
        
        # Train meta-model if using stacking
        if self.ensemble_method == 'stacking' and self.meta_model:
            self.meta_model.train(ensemble_predictions, y)
        
        self.is_trained = True
        
        # Calculate ensemble metrics
        if validation_data:
            val_ensemble_predictions = self._get_ensemble_predictions(validation_data[0])
            
            if self.meta_model and self.ensemble_method == 'stacking':
                final_predictions = self.meta_model.predict(val_ensemble_predictions)
            else:
                final_predictions = self._combine_predictions(val_ensemble_predictions)
            
            val_score = self._calculate_score(validation_data[1], final_predictions)
            
            return {
                'base_model_scores': base_model_scores,
                'ensemble_val_score': val_score,
                'n_base_models': len(self.base_models)
            }
        
        return {
            'base_model_scores': base_model_scores,
            'n_base_models': len(self.base_models)
        }
    
    def _get_ensemble_predictions(self, X: np.ndarray) -> np.ndarray:
        """Get predictions from all base models"""
        predictions = []
        
        for base_model in self.base_models:
            if base_model.is_trained:
                pred = base_model.predict(X)
                predictions.append(pred.reshape(-1, 1))
        
        if not predictions:
            raise ValueError("No trained base models available")
        
        return np.hstack(predictions)
    
    def _combine_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """Combine predictions from base models"""
        if self.ensemble_method == 'voting':
            # Majority voting for classification
            if self.model_type == 'classification':
                # Use scipy stats mode for majority voting
                from scipy import stats
                return stats.mode(predictions, axis=1)[0].flatten()
            # Average for regression
            else:
                return np.mean(predictions, axis=1)
        elif self.ensemble_method == 'averaging':
            return np.mean(predictions, axis=1)
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
    
    def _calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate performance score"""
        if self.model_type == 'classification':
            return accuracy_score(y_true, y_pred)
        else:
            return -mean_squared_error(y_true, y_pred)  # Negative MSE for consistency
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions"""
        if not self.is_trained:
            raise ValueError("Ensemble model is not trained")
        
        # Get base model predictions
        ensemble_predictions = self._get_ensemble_predictions(X)
        
        # Combine predictions
        if self.meta_model and self.ensemble_method == 'stacking':
            return self.meta_model.predict(ensemble_predictions)
        else:
            return self._combine_predictions(ensemble_predictions)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Make probability predictions"""
        if not self.is_trained:
            raise ValueError("Ensemble model is not trained")
        
        # Average probabilities from all models that support it
        probabilities = []
        
        for base_model in self.base_models:
            if hasattr(base_model, 'predict_proba') and base_model.is_trained:
                proba = base_model.predict_proba(X)
                probabilities.append(proba)
        
        if not probabilities:
            raise ValueError("No base models support probability prediction")
        
        # Average probabilities
        return np.mean(probabilities, axis=0)

class VotingEnsemble(EnsembleModel):
    """Voting ensemble model"""
    
    def __init__(self, model_name: str, model_type: str, voting_type: str = 'soft', **kwargs):
        super().__init__(model_name, model_type, **kwargs)
        self.ensemble_method = 'voting'
        self.voting_type = voting_type

class StackingEnsemble(EnsembleModel):
    """Stacking ensemble model"""
    
    def __init__(self, model_name: str, model_type: str, meta_model: Optional[BaseModel] = None, **kwargs):
        super().__init__(model_name, model_type, **kwargs)
        self.ensemble_method = 'stacking'
        self.meta_model = meta_model or self._create_default_meta_model()
    
    def _create_default_meta_model(self) -> BaseModel:
        """Create default meta-model for stacking"""
        if self.model_type == 'classification':
            from ..base import SklearnModel
            
            class LogisticMetaModel(SklearnModel):
                def __init__(self):
                    super().__init__("logistic_meta", "meta")
                    from sklearn.linear_model import LogisticRegression
                    self.model = LogisticRegression(random_state=42)
            
            return LogisticMetaModel()
        else:
            from ..base import SklearnModel
            
            class LinearMetaModel(SklearnModel):
                def __init__(self):
                    super().__init__("linear_meta", "meta")
                    from sklearn.linear_model import LinearRegression
                    self.model = LinearRegression()
            
            return LinearMetaModel()

class BlendingEnsemble(EnsembleModel):
    """Blending ensemble model (simplified stacking)"""
    
    def __init__(self, model_name: str, model_type: str, blend_ratio: float = 0.7, **kwargs):
        super().__init__(model_name, model_type, **kwargs)
        self.ensemble_method = 'blending'
        self.blend_ratio = blend_ratio
        self.holdout_size = 1.0 - blend_ratio
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
              **kwargs) -> Dict[str, float]:
        """Train blending ensemble with holdout set"""
        
        if not self.base_models:
            raise ValueError("No base models added to ensemble")
        
        # Split data for blending
        from sklearn.model_selection import train_test_split
        X_blend, X_holdout, y_blend, y_holdout = train_test_split(
            X, y, test_size=self.holdout_size, random_state=42
        )
        
        # Train base models on blend set
        base_model_scores = {}
        for base_model in self.base_models:
            score = base_model.train(X_blend, y_blend)
            base_model_scores[base_model.model_name] = score
        
        # Generate predictions on holdout set for meta-model
        holdout_predictions = self._get_ensemble_predictions(X_holdout)
        
        # Train meta-model on holdout predictions
        if self.meta_model:
            self.meta_model.train(holdout_predictions, y_holdout)
        
        self.is_trained = True
        
        return {
            'base_model_scores': base_model_scores,
            'holdout_size': len(X_holdout),
            'blend_size': len(X_blend)
        }

# Convenience functions
def create_voting_ensemble(model_name: str, base_models: List[BaseModel], 
                          model_type: str, voting_type: str = 'soft') -> VotingEnsemble:
    """Create a voting ensemble"""
    ensemble = VotingEnsemble(model_name, model_type, voting_type)
    
    for base_model in base_models:
        ensemble.add_base_model(base_model)
    
    return ensemble

def create_stacking_ensemble(model_name: str, base_models: List[BaseModel], 
                           model_type: str, meta_model: Optional[BaseModel] = None) -> StackingEnsemble:
    """Create a stacking ensemble"""
    ensemble = StackingEnsemble(model_name, model_type, meta_model)
    
    for base_model in base_models:
        ensemble.add_base_model(base_model)
    
    return ensemble

# Example usage
if __name__ == "__main__":
    from ..traditional.xgboost_model import XGBoostFootballModel
    from ..traditional.lightgbm_model import LightGBMFootballModel
    
    # Create sample data
    X_sample = np.random.randn(1000, 50)
    y_sample = np.random.randint(0, 3, 1000)
    
    # Create base models
    xgb_model = XGBoostFootballModel("xgb_base", "classification")
    lgb_model = LightGBMFootballModel("lgb_base", "classification")
    
    # Train base models
    xgb_model.train(X_sample, y_sample)
    lgb_model.train(X_sample, y_sample)
    
    # Create voting ensemble
    voting_ensemble = create_voting_ensemble(
        "voting_football", 
        [xgb_model, lgb_model], 
        "classification"
    )
    
    # Train ensemble
    results = voting_ensemble.train(X_sample, y_sample)
    print(f"Voting ensemble results: {results}")
    
    # Make predictions
    test_X = np.random.randn(10, 50)
    predictions = voting_ensemble.predict(test_X)
    print(f"Ensemble predictions: {predictions}")
EOF

    log "✅ Ensemble Models erstellt"
}

# Create model registry
create_model_registry() {
    log "📋 Erstelle Model Registry..."
    
    cat > ~/football-ai-system/backend/models/registry.py << 'EOF'
"""
Model Registry for managing ML models
"""

from typing import Dict, List, Optional, Any, Type
import json
from pathlib import Path
import logging
from datetime import datetime
import joblib

from .base import BaseModel
from .neural_nets.correct_score_model import CorrectScoreModel
from .neural_nets.over_under_model import OverUnderModel
from .traditional.xgboost_model import XGBoostFootballModel
from .traditional.lightgbm_model import LightGBMFootballModel
from .traditional.catboost_model import CatBoostFootballModel
from .ensemble.ensemble_models import VotingEnsemble, StackingEnsemble

logger = logging.getLogger(__name__)

class ModelRegistry:
    """Central registry for managing ML models"""
    
    def __init__(self, registry_path: str = "./models/registry.json"):
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.models: Dict[str, BaseModel] = {}
        self.model_metadata: Dict[str, Dict[str, Any]] = {}
        self.model_performance: Dict[str, Dict[str, float]] = {}
        
        # Model class mapping
        self.model_classes = {
            'correct_score': CorrectScoreModel,
            'over_under': OverUnderModel,
            'xgboost': XGBoostFootballModel,
            'lightgbm': LightGBMFootballModel,
            'catboost': CatBoostFootballModel,
            'voting_ensemble': VotingEnsemble,
            'stacking_ensemble': StackingEnsemble
        }
        
        # Load existing registry
        self.load_registry()
    
    def register_model(self, model: BaseModel, **metadata) -> str:
        """Register a model in the registry"""
        
        model_id = f"{model.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Store model
        self.models[model_id] = model
        
        # Store metadata
        self.model_metadata[model_id] = {
            'model_name': model.model_name,
            'model_type': model.model_type,
            'model_class': model.__class__.__name__,
            'registered_at': datetime.now().isoformat(),
            'is_trained': model.is_trained,
            'metadata': metadata
        }
        
        logger.info(f"✅ Model registered: {model_id}")
        return model_id
    
    def get_model(self, model_id: str) -> Optional[BaseModel]:
        """Get a model by ID"""
        return self.models.get(model_id)
    
    def list_models(self, model_type: Optional[str] = None) -> List[str]:
        """List all registered models"""
        if model_type:
            return [
                model_id for model_id, metadata in self.model_metadata.items()
                if metadata['model_type'] == model_type
            ]
        return list(self.models.keys())
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model information"""
        if model_id not in self.model_metadata:
            return None
        
        info = self.model_metadata[model_id].copy()
        
        # Add model-specific information
        if model_id in self.models:
            model = self.models[model_id]
            info.update(model.get_model_info())
        
        # Add performance metrics
        if model_id in self.model_performance:
            info['performance'] = self.model_performance[model_id]
        
        return info
    
    def update_model_performance(self, model_id: str, metrics: Dict[str, float]) -> None:
        """Update model performance metrics"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found in registry")
        
        self.model_performance[model_id] = metrics
        self.model_metadata[model_id]['last_updated'] = datetime.now().isoformat()
        
        logger.info(f"📊 Performance updated for model: {model_id}")
    
    def save_model(self, model_id: str, filepath: str) -> None:
        """Save a model to disk"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found in registry")
        
        model = self.models[model_id]
        model.save_model(filepath)
        
        # Update registry
        self.model_metadata[model_id]['saved_at'] = datetime.now().isoformat()
        self.model_metadata[model_id]['filepath'] = filepath
        
        logger.info(f"💾 Model saved: {model_id} -> {filepath}")
    
    def load_model(self, model_id: str, filepath: str, model_class: str) -> None:
        """Load a model from disk"""
        if model_class not in self.model_classes:
            raise ValueError(f"Unknown model class: {model_class}")
        
        # Create model instance
        model_class_obj = self.model_classes[model_class]
        model = model_class_obj(f"loaded_{model_class}", model_class)
        
        # Load model data
        model.load_model(filepath)
        
        # Register loaded model
        loaded_model_id = self.register_model(model, source='loaded_from_disk', filepath=filepath)
        
        logger.info(f"📂 Model loaded: {filepath} -> {loaded_model_id}")
    
    def create_model(self, model_class: str, model_name: str, **kwargs) -> str:
        """Create and register a new model"""
        if model_class not in self.model_classes:
            raise ValueError(f"Unknown model class: {model_class}")
        
        model_class_obj = self.model_classes[model_class]
        model = model_class_obj(model_name, model_class, **kwargs)
        
        return self.register_model(model, creation_method='factory')
    
    def delete_model(self, model_id: str) -> None:
        """Delete a model from registry"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found in registry")
        
        # Remove model and metadata
        del self.models[model_id]
        del self.model_metadata[model_id]
        
        if model_id in self.model_performance:
            del self.model_performance[model_id]
        
        logger.info(f"🗑️ Model deleted from registry: {model_id}")
    
    def save_registry(self) -> None:
        """Save registry to disk"""
        registry_data = {
            'models': self.model_metadata,
            'performance': self.model_performance,
            'saved_at': datetime.now().isoformat()
        }
        
        with open(self.registry_path, 'w') as f:
            json.dump(registry_data, f, indent=2, default=str)
        
        logger.info(f"📋 Registry saved to {self.registry_path}")
    
    def load_registry(self) -> None:
        """Load registry from disk"""
        if not self.registry_path.exists():
            logger.info("No existing registry found, starting fresh")
            return
        
        try:
            with open(self.registry_path, 'r') as f:
                registry_data = json.load(f)
            
            self.model_metadata = registry_data.get('models', {})
            self.model_performance = registry_data.get('performance', {})
            
            logger.info(f"📋 Registry loaded from {self.registry_path}")
            
        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
    
    def get_best_model(self, metric: str = 'accuracy', model_type: Optional[str] = None) -> Optional[str]:
        """Get the best performing model"""
        models_to_consider = self.list_models(model_type)
        
        if not models_to_consider:
            return None
        
        best_model_id = None
        best_score = float('-inf')
        
        for model_id in models_to_consider:
            if model_id in self.model_performance:
                performance = self.model_performance[model_id]
                if metric in performance:
                    score = performance[metric]
                    if score > best_score:
                        best_score = score
                        best_model_id = model_id
        
        return best_model_id
    
    def compare_models(self, model_ids: List[str], metrics: List[str]) -> Dict[str, Dict[str, float]]:
        """Compare multiple models"""
        comparison = {}
        
        for model_id in model_ids:
            if model_id in self.model_performance:
                comparison[model_id] = {
                    metric: self.model_performance[model_id].get(metric, None)
                    for metric in metrics
                }
        
        return comparison

# Global registry instance
model_registry = ModelRegistry()

# Convenience functions
def register_model(model: BaseModel, **metadata) -> str:
    """Register a model in the global registry"""
    return model_registry.register_model(model, **metadata)

def get_model(model_id: str) -> Optional[BaseModel]:
    """Get a model from the global registry"""
    return model_registry.get_model(model_id)

def list_models(model_type: Optional[str] = None) -> List[str]:
    """List models in the global registry"""
    return model_registry.list_models(model_type)

def get_best_model(metric: str = 'accuracy', model_type: Optional[str] = None) -> Optional[str]:
    """Get the best model from the global registry"""
    return model_registry.get_best_model(metric, model_type)

# Example usage
if __name__ == "__main__":
    # Create some models
    registry = ModelRegistry()
    
    # Create a simple model
    from .neural_nets.correct_score_model import CorrectScoreModel
    model = CorrectScoreModel("test_model", input_dim=100)
    
    # Register model
    model_id = registry.register_model(model, test_data="sample")
    print(f"Registered model: {model_id}")
    
    # List models
    models = registry.list_models()
    print(f"Available models: {models}")
    
    # Get model info
    info = registry.get_model_info(model_id)
    print(f"Model info: {info}")
    
    # Save registry
    registry.save_registry()
EOF

    log "✅ Model Registry erstellt"
}

# Create training scripts
create_training_scripts() {
    log "🏋️ Erstelle Training Scripts..."
    
    cat > ~/football-ai-system/scripts/train_models.py << 'EOF'
#!/usr/bin/env python3
"""
Model Training Script for Football AI System
"""

import argparse
import sys
from pathlib import Path
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch

# Add backend to path
backend_dir = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from models.neural_nets.correct_score_model import CorrectScoreModel
from models.neural_nets.over_under_model import OverUnderModel
from models.traditional.xgboost_model import XGBoostFootballModel
from models.traditional.lightgbm_model import LightGBMFootballModel
from models.ensemble.ensemble_models import create_voting_ensemble
from models.registry import model_registry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_data(n_samples: int = 10000, n_features: int = 100) -> tuple:
    """Create sample training data"""
    
    logger.info(f"Creating sample data: {n_samples} samples, {n_features} features")
    
    # Generate random features
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    
    # Generate targets based on model type
    # Correct Score targets
    home_goals = np.random.randint(0, 6, n_samples)
    away_goals = np.random.randint(0, 6, n_samples)
    outcome = np.where(home_goals > away_goals, 0, np.where(home_goals == away_goals, 1, 2))
    total_goals = home_goals + away_goals
    
    correct_score_targets = {
        'home_goals': home_goals,
        'away_goals': away_goals,
        'outcome': outcome,
        'total_goals': total_goals
    }
    
    # Over/Under targets
    over_under_targets = total_goals
    
    # Classification targets
    match_winner = outcome
    
    return X, correct_score_targets, over_under_targets, match_winner

def train_correct_score_model(X: np.ndarray, y: dict, epochs: int = 50) -> str:
    """Train correct score neural network"""
    
    logger.info("Training Correct Score Neural Network...")
    
    # Create model
    model = CorrectScoreModel(
        model_name="correct_score_v1",
        input_dim=X.shape[1],
        hidden_dims=[512, 256, 128],
        dropout_rate=0.3,
        learning_rate=0.001
    )
    
    # Split data
    X_train, X_val, y_train_home, y_val_home = train_test_split(
        X, y['home_goals'], test_size=0.2, random_state=42
    )
    _, _, y_train_away, y_val_away = train_test_split(
        X, y['away_goals'], test_size=0.2, random_state=42
    )
    _, _, y_train_outcome, y_val_outcome = train_test_split(
        X, y['outcome'], test_size=0.2, random_state=42
    )
    _, _, y_train_total, y_val_total = train_test_split(
        X, y['total_goals'], test_size=0.2, random_state=42
    )
    
    # Prepare training targets
    y_train = {
        'home_goals': y_train_home,
        'away_goals': y_train_away,
        'outcome': y_train_outcome,
        'total_goals': y_train_total
    }
    
    # Train model
    results = model.train(X_train, y_train, epochs=epochs, batch_size=64)
    
    # Register model
    model_id = model_registry.register_model(
        model,
        model_architecture="neural_network",
        training_data_size=len(X),
        epochs_trained=epochs,
        **results
    )
    
    # Save model
    model_path = Path(f"models/neural_nets/correct_score/{model_id}.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))
    
    logger.info(f"✅ Correct Score Model trained and registered: {model_id}")
    return model_id

def train_over_under_model(X: np.ndarray, y: np.ndarray, epochs: int = 50) -> str:
    """Train over/under neural network"""
    
    logger.info("Training Over/Under Neural Network...")
    
    # Create model
    model = OverUnderModel(
        model_name="over_under_v1",
        input_dim=X.shape[1],
        hidden_dims=[256, 128, 64],
        dropout_rate=0.3,
        learning_rate=0.001
    )
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    results = model.train(X_train, y_train, epochs=epochs, batch_size=64)
    
    # Register model
    model_id = model_registry.register_model(
        model,
        model_architecture="neural_network",
        training_data_size=len(X),
        epochs_trained=epochs,
        **results
    )
    
    # Save model
    model_path = Path(f"models/neural_nets/over_under/{model_id}.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))
    
    logger.info(f"✅ Over/Under Model trained and registered: {model_id}")
    return model_id

def train_xgboost_model(X: np.ndarray, y: np.ndarray, model_type: str = 'classification') -> str:
    """Train XGBoost model"""
    
    logger.info(f"Training XGBoost {model_type} model...")
    
    # Create model
    model = XGBoostFootballModel(
        model_name=f"xgboost_{model_type}_v1",
        model_type=model_type,
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.1
    )
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    results = model.train(X_train, y_train, validation_data=(X_val, y_val))
    
    # Register model
    model_id = model_registry.register_model(
        model,
        model_type="gradient_boosting",
        framework="xgboost",
        training_data_size=len(X),
        **results
    )
    
    # Save model
    model_path = Path(f"models/traditional/xgboost/{model_id}.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))
    
    logger.info(f"✅ XGBoost Model trained and registered: {model_id}")
    return model_id

def train_lightgbm_model(X: np.ndarray, y: np.ndarray, model_type: str = 'classification') -> str:
    """Train LightGBM model"""
    
    logger.info(f"Training LightGBM {model_type} model...")
    
    # Create model
    model = LightGBMFootballModel(
        model_name=f"lightgbm_{model_type}_v1",
        model_type=model_type,
        n_estimators=1000,
        max_depth=-1,
        learning_rate=0.1
    )
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    results = model.train(X_train, y_train, validation_data=(X_val, y_val))
    
    # Register model
    model_id = model_registry.register_model(
        model,
        model_type="gradient_boosting",
        framework="lightgbm",
        training_data_size=len(X),
        **results
    )
    
    # Save model
    model_path = Path(f"models/traditional/lightgbm/{model_id}.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))
    
    logger.info(f"✅ LightGBM Model trained and registered: {model_id}")
    return model_id

def train_ensemble_models(model_ids: List[str]) -> str:
    """Train ensemble model"""
    
    logger.info("Training Ensemble Model...")
    
    # Get trained models
    base_models = []
    for model_id in model_ids:
        model = model_registry.get_model(model_id)
        if model and model.is_trained:
            base_models.append(model)
    
    if len(base_models) < 2:
        logger.warning("Need at least 2 trained models for ensemble")
        return None
    
    # Create voting ensemble
    ensemble = create_voting_ensemble(
        model_name="voting_ensemble_v1",
        base_models=base_models,
        model_type="classification",
        voting_type="soft"
    )
    
    # Create sample data for ensemble training
    X, _, _, y = create_sample_data(1000, 50)
    
    # Train ensemble
    results = ensemble.train(X, y)
    
    # Register ensemble
    ensemble_id = model_registry.register_model(
        ensemble,
        ensemble_type="voting",
        base_models=model_ids,
        **results
    )
    
    # Save ensemble
    ensemble_path = Path(f"models/ensemble/voting/{ensemble_id}.pkl")
    ensemble_path.parent.mkdir(parents=True, exist_ok=True)
    ensemble.save_model(str(ensemble_path))
    
    logger.info(f"✅ Ensemble Model trained and registered: {ensemble_id}")
    return ensemble_id

def main():
    """Main training function"""
    
    parser = argparse.ArgumentParser(description="Train Football AI Models")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--sample-size", type=int, default=10000, help="Sample data size")
    parser.add_argument("--features", type=int, default=100, help="Number of features")
    parser.add_argument("--models", nargs="+", default=["all"], 
                       choices=["all", "correct_score", "over_under", "xgboost", "lightgbm", "ensemble"],
                       help="Models to train")
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting Model Training...")
    logger.info(f"Configuration: {args}")
    
    # Create sample data
    X, correct_score_y, over_under_y, match_winner_y = create_sample_data(
        args.sample_size, args.features
    )
    
    trained_models = []
    
    # Train requested models
    if "all" in args.models or "correct_score" in args.models:
        model_id = train_correct_score_model(X, correct_score_y, args.epochs)
        trained_models.append(model_id)
    
    if "all" in args.models or "over_under" in args.models:
        model_id = train_over_under_model(X, over_under_y, args.epochs)
        trained_models.append(model_id)
    
    if "all" in args.models or "xgboost" in args.models:
        model_id = train_xgboost_model(X, match_winner_y, "classification")
        trained_models.append(model_id)
    
    if "all" in args.models or "lightgbm" in args.models:
        model_id = train_lightgbm_model(X, match_winner_y, "classification")
        trained_models.append(model_id)
    
    if "all" in args.models or "ensemble" in args.models:
        if len(trained_models) >= 2:
            ensemble_id = train_ensemble_models(trained_models[:2])  # Use first 2 models
            if ensemble_id:
                trained_models.append(ensemble_id)
    
    # Save registry
    model_registry.save_registry()
    
    logger.info("✅ Model Training completed!")
    logger.info(f"Trained models: {trained_models}")
    
    # Display summary
    print("\n" + "="*60)
    print("🏆 TRAINING SUMMARY")
    print("="*60)
    
    for model_id in trained_models:
        info = model_registry.get_model_info(model_id)
        if info:
            print(f"📊 Model: {info['model_name']}")
            print(f"   Type: {info['model_type']}")
            print(f"   Trained: {info.get('is_trained', False)}")
            print(f"   ID: {model_id}")
            print()

if __name__ == "__main__":
    main()
EOF

    chmod +x ~/football-ai-system/scripts/train_models.py
    
    log "✅ Training Scripts erstellt"
}

# Main execution
main() {
    check_user
    activate_env
    
    log "🚀 Starte ML Models Setup (RTX 3090 Optimized)"
    
    # Create model structures
    create_model_bases
    create_neural_networks
    create_traditional_models
    create_ensemble_models
    create_model_registry
    create_training_scripts
    
    # Create additional model files
    mkdir -p ~/football-ai-system/backend/models/neural_nets
    mkdir -p ~/football-ai-system/backend/models/traditional
    mkdir -p ~/football-ai-system/backend/models/ensemble
    
    # Create __init__.py files
    touch ~/football-ai-system/backend/models/__init__.py
    touch ~/football-ai-system/backend/models/neural_nets/__init__.py
    touch ~/football-ai-system/backend/models/traditional/__init__.py
    touch ~/football-ai-system/backend/models/ensemble/__init__.py
    
    log "✅ ML Models Setup abgeschlossen!"
    log "🎯 Nächster Schritt: ./04-frontend-setup.sh ausführen"
    log "🏋️ Optional: python scripts/train_models.py --help"
}

# Execute main function
main "$@"
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Sabit değişkenler
SEQUENCE_LENGTH = 96  # 24 saatlik veri kullanacağız
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.0001

# Sınıflandırma eşikleri
THRESHOLDS = {
    'very_large_drop': -1.0,    # -1.0'den düşük
    'large_drop': -0.6,         # -1.0 ile -0.6 arası
    'medium_drop': -0.2,        # -0.6 ile -0.2 arası
    'small_drop': 0,            # -0.2 ile 0 arası
    'small_rise': 0.2,          # 0 ile 0.2 arası
    'medium_rise': 0.6,         # 0.2 ile 0.6 arası
    'large_rise': 1.0,          # 0.6 ile 1.0 arası
    # 1.0'den büyük çok büyük yükseliş
}

def classify_movement(percent_change):
    """
    Yüzdesel değişime göre hareketin sınıfını belirler
    Returns: 0-7 arası bir sınıf numarası
    """
    if percent_change < THRESHOLDS['very_large_drop']:
        return 0  # Çok büyük düşüş
    elif percent_change < THRESHOLDS['large_drop']:
        return 1  # Büyük düşüş
    elif percent_change < THRESHOLDS['medium_drop']:
        return 2  # Orta düşüş
    elif percent_change < THRESHOLDS['small_drop']:
        return 3  # Küçük düşüş
    elif percent_change < THRESHOLDS['small_rise']:
        return 4  # Küçük yükseliş
    elif percent_change < THRESHOLDS['medium_rise']:
        return 5  # Orta yükseliş
    elif percent_change < THRESHOLDS['large_rise']:
        return 6  # Büyük yükseliş
    else:
        return 7  # Çok büyük yükseliş

class CandlestickDataset(Dataset):
    def __init__(self, data, sequence_length=SEQUENCE_LENGTH):
        self.data = data
        self.sequence_length = sequence_length
        
        # Öznitelik oluşturma
        self.features = self._create_features()
        
        # Etiketleri oluştur
        self.labels = self._create_labels()
        
        # Sadece çok büyük yükseliş veya düşüşten sonraki verileri seç
        self.valid_indices = self._find_valid_sequences()
        
        # Standard scaler ile normalizasyon
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)
        
        # Sequence oluşturma
        self.X, self.y = self._create_sequences()
    
    def _find_valid_sequences(self):
        """Çok büyük veya büyük yükseliş/düşüşten sonraki indeksleri bul"""
        valid_indices = []
        for i in range(len(self.features) - self.sequence_length):
            last_idx = i + self.sequence_length - 1
            if self.labels[last_idx] in [0, 1, 6, 7]:  # Çok büyük/büyük yükseliş veya düşüş
                valid_indices.append(i)
        return valid_indices
    
    def _create_features(self):
        df = self.data.copy()
        
        features = pd.DataFrame()
        features['open_close_ratio'] = (df['Close'] - df['Open']) / df['Open'] * 100
        features['high_low_ratio'] = (df['High'] - df['Low']) / df['Low'] * 100
        features['volume'] = df['Volume']
        features['hour'] = df['Hour']
        features['price_momentum'] = df['Close'].pct_change()
        
        return features.fillna(0)
    
    def _create_labels(self):
        df = self.data.copy()
        changes = ((df['Close'] - df['Open']) / df['Open'] * 100).values
        return np.array([classify_movement(change) for change in changes])
    
    def _create_sequences(self):
        X, y = [], []
        for i in self.valid_indices:
            X.append(self.features[i:i+self.sequence_length])
            y.append(self.labels[i+self.sequence_length])
        return np.array(X), np.array(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.LongTensor([self.y[idx]])[0]

class PricePredictionLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=128, num_layers=2, output_size=8):
        super(PricePredictionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def calculate_binary_accuracy(predictions, labels):
    """
    Long/Short tahmini için ikili doğruluk hesaplama
    """
    # Düşüş ve yükseliş olasılıklarını topla
    down_probs = predictions[:, :4].sum(dim=1)
    up_probs = predictions[:, 4:].sum(dim=1)
    
    # Binary tahminler
    binary_preds = (up_probs > down_probs).long()
    binary_labels = (labels >= 4).long()
    
    return (binary_preds == binary_labels).float().mean()

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device):
    """
    Model eğitim fonksiyonu
    """
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        binary_accuracies = []
        class_accuracies = []
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 50)
        
        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, labels)
            
            # Backward pass ve optimizasyon
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Metrikleri hesapla
            total_loss += loss.item()
            binary_acc = calculate_binary_accuracy(torch.softmax(outputs, dim=1), labels)
            class_acc = (outputs.argmax(dim=1) == labels).float().mean()
            
            binary_accuracies.append(binary_acc.item())
            class_accuracies.append(class_acc.item())
            
            # Her 10 batch'te bir ilerlemeyi göster
            if (batch_idx + 1) % 10 == 0:
                print(f"Batch {batch_idx+1}: Loss: {loss.item():.4f}, "
                      f"Binary Acc: {binary_acc:.4f}, "
                      f"Class Acc: {class_acc:.4f}")
        
        # Epoch sonuçlarını yazdır
        avg_loss = total_loss / len(train_loader)
        avg_binary_acc = sum(binary_accuracies) / len(binary_accuracies)
        avg_class_acc = sum(class_accuracies) / len(class_accuracies)
        
        print(f"\nEpoch {epoch+1} Results:")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Average Binary Accuracy: {avg_binary_acc:.4f}")
        print(f"Average Class Accuracy: {avg_class_acc:.4f}")
        
        # Validation
        val_binary_acc, val_class_acc = evaluate_model(model, val_loader, device)
        print(f"Validation Binary Accuracy: {val_binary_acc:.4f}")
        print(f"Validation Class Accuracy: {val_class_acc:.4f}")
        
        # En iyi modeli kaydet
        if val_binary_acc > best_val_acc:
            best_val_acc = val_binary_acc
            torch.save(model.state_dict(), "64500kucukaz.pth")
            print("Model kaydedildi: mumlu.pth")

def evaluate_model(model, loader, device):
    """
    Model değerlendirme fonksiyonu
    """
    model.eval()
    binary_accuracies = []
    class_accuracies = []
    
    with torch.no_grad():
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            
            binary_acc = calculate_binary_accuracy(torch.softmax(outputs, dim=1), labels)
            class_acc = (outputs.argmax(dim=1) == labels).float().mean()
            
            binary_accuracies.append(binary_acc.item())
            class_accuracies.append(class_acc.item())
    
    avg_binary_acc = sum(binary_accuracies) / len(binary_accuracies)
    avg_class_acc = sum(class_accuracies) / len(class_accuracies)
    
    return avg_binary_acc, avg_class_acc

def predict_next_move(model, data, device):
    """
    Sonraki hareket için tahmin yapma fonksiyonu
    Returns: (tahmin_sınıfı, güven_skoru, yön)
    """
    model.eval()
    with torch.no_grad():
        output = model(data.to(device))
        probabilities = torch.softmax(output, dim=1)
        
        # Düşüş ve yükseliş olasılıklarını hesapla
        down_prob = probabilities[0, :4].sum().item()
        up_prob = probabilities[0, 4:].sum().item()
        
        # En yüksek olasılıklı sınıfı bul
        pred_class = output.argmax(dim=1).item()
        
        # Yön ve güven skoru belirle
        if up_prob > down_prob:
            confidence = up_prob
            direction = "LONG"
        else:
            confidence = down_prob
            direction = "SHORT"
            
        return pred_class, confidence, direction
    
def load_and_preprocess_data(file_path):
    """
    Veri yükleme ve ön işleme fonksiyonu
    """
    # CSV dosyasını oku
    df = pd.read_csv(file_path)
    
    # Veriyi eğitim ve validasyon için böl
    train_data, val_data = train_test_split(df, test_size=0.2, shuffle=False)
    
    # Dataset'leri oluştur
    train_dataset = CandlestickDataset(train_data)
    val_dataset = CandlestickDataset(val_data)
    
    # DataLoader'ları oluştur
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    return train_loader, val_loader

def main():
    """
    Ana fonksiyon
    """
    # Cihazı belirle
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan cihaz: {device}")
    
    # Veriyi yükle
    print("Veri yükleniyor...")
    train_loader, val_loader = load_and_preprocess_data("ilk64500kisa.csv")
    
    # Modeli oluştur
    print("Model oluşturuluyor...")
    model = PricePredictionLSTM().to(device)
    
    # Loss fonksiyonu ve optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Modeli eğit
    print("Eğitim başlıyor...")
    train_model(model, train_loader, val_loader, criterion, optimizer, EPOCHS, device)
    
    print("\nEğitim tamamlandı!")
    print("En iyi model 'mumlu.pth' olarak kaydedildi.")

if __name__ == "__main__":
    main()
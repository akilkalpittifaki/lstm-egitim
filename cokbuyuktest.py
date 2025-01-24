import pandas as pd
import numpy as np
import torch
from lstmmum import PricePredictionLSTM, CandlestickDataset, classify_movement
import matplotlib.pyplot as plt
import seaborn as sns

def test_model(model_path, test_data_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan cihaz: {device}")
    
    print("Test verisi yükleniyor...")
    df = pd.read_csv(test_data_path)
    test_dataset = CandlestickDataset(df)
    
    # Modeli yükle
    model = PricePredictionLSTM().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Tahmin sonuçlarını tutacak listeler
    predictions = []
    
    # Her sequence için tahmin yap
    sequence_length = 96
    for i in range(len(df) - sequence_length):
        # Önceki mumun sınıfını belirle
        prev_idx = i + sequence_length - 1
        prev_change = ((df['Close'].iloc[prev_idx] - df['Open'].iloc[prev_idx]) / 
                      df['Open'].iloc[prev_idx] * 100)
        prev_class = classify_movement(prev_change)
        
        # Sadece çok büyük yükseliş (7) veya çok büyük düşüşten (0) sonra tahmin yap
        if prev_class not in [0, 7]:
            continue
            
        # Sequence'i al
        data = torch.FloatTensor(test_dataset.X[i]).unsqueeze(0)
        
        # Tahmin yap
        with torch.no_grad():
            output = model(data.to(device))
            probabilities = torch.softmax(output, dim=1)
            
            down_prob = probabilities[0, :4].sum().item()
            up_prob = probabilities[0, 4:].sum().item()
            
            confidence = max(up_prob, down_prob)
            predicted_direction = "YÜKSELİŞ" if up_prob > down_prob else "DÜŞÜŞ"
            
            actual_idx = i + sequence_length
            actual_change = ((df['Close'].iloc[actual_idx] - df['Open'].iloc[actual_idx]) / 
                           df['Open'].iloc[actual_idx] * 100)
            actual_direction = "YÜKSELİŞ" if actual_change >= 0 else "DÜŞÜŞ"
            actual_class = classify_movement(actual_change)
            
            # Detaylı tahmin bilgilerini kaydet
            predictions.append({
                'mum_no': actual_idx + 1,
                'prev_class': prev_class,
                'open': df['Open'].iloc[actual_idx],
                'close': df['Close'].iloc[actual_idx],
                'predicted': predicted_direction,
                'actual': actual_direction,
                'confidence': confidence,
                'is_correct': predicted_direction == actual_direction,
                'price_change': actual_change,
                'actual_class': actual_class,
                'down_probability': down_prob,
                'up_probability': up_prob
            })
    
    # Mum sınıfı dağılımını hesapla
    all_candles = []
    for i in range(len(df)):
        change = ((df['Close'].iloc[i] - df['Open'].iloc[i]) / df['Open'].iloc[i] * 100)
        all_candles.append(classify_movement(change))
    
    candle_distribution = pd.Series(all_candles).value_counts().sort_index()
    
    print("\n=== MUM SINIFI DAĞILIMI ===")
    class_names = ["Çok Büyük Düşüş", "Büyük Düşüş", "Orta Düşüş", "Küçük Düşüş",
                  "Küçük Yükseliş", "Orta Yükseliş", "Büyük Yükseliş", "Çok Büyük Yükseliş"]
    
    for class_idx, count in candle_distribution.items():
        print(f"{class_names[class_idx]}: {count} mum")

    # Sonuçları DataFrame'e dönüştür
    results_df = pd.DataFrame(predictions)
    
    if len(results_df) == 0:
        print("\nHiç tahmin yapılmadı!")
        return None, 0, {}
        
    # Genel başarı oranını hesapla
    overall_accuracy = results_df['is_correct'].mean() * 100
    
    # Farklı güven skorlarına göre başarı oranları
    confidence_thresholds = [0.51, 0.52, 0.53, 0.54, 0.55]
    accuracy_by_confidence = {}
    
    for threshold in confidence_thresholds:
        filtered_predictions = results_df[results_df['confidence'] >= threshold]
        if len(filtered_predictions) > 0:
            accuracy = filtered_predictions['is_correct'].mean() * 100
            accuracy_by_confidence[threshold] = {
                'accuracy': accuracy,
                'signal_count': len(filtered_predictions)
            }

    # Önceki mum sınıfına göre başarı analizi
    print("\n=== ÖNCEKİ MUM SINIFINA GÖRE BAŞARI ANALİZİ ===")
    for prev_class in [0, 7]:
        class_preds = results_df[results_df['prev_class'] == prev_class]
        if len(class_preds) > 0:
            accuracy = class_preds['is_correct'].mean() * 100
            print(f"\n{class_names[prev_class]}ten Sonra:")
            print(f"Tahmin Sayısı: {len(class_preds)}")
            print(f"Başarı Oranı: %{accuracy:.2f}")

    print("\n=== ÖZET BAŞARI ANALİZİ ===")
    print(f"\nGenel Başarı Oranı: %{overall_accuracy:.2f}")
    print(f"Toplam Tahmin Sayısı: {len(results_df)}")
    
    return results_df, overall_accuracy, accuracy_by_confidence

if __name__ == "__main__":
    model_path = "64500mumlu.pth"
    test_data_path = "kisa64500.csv"
    results_df, overall_accuracy, accuracy_by_confidence = test_model(model_path, test_data_path)
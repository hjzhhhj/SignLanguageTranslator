import json
from sklearn.model_selection import train_test_split
from src.sign_language_model import SignLanguageModel
from src.data_collector import SignLanguageDataCollector

def train_sign_language_model():
    # 수어 인식 모델 학습 및 평가, 모델/기록/그래프 저장
    print("모델 학습 준비 중...")
    
    # 1. 데이터 수집기 초기화
    # 데이터 수집기에는 저장된 랜드마크 데이터들을 로드하고 가공하는 기능이 있음
    collector = SignLanguageDataCollector()
    
    # 2. 학습 데이터 로드
    print("학습 데이터 로드 중...")
    X_train, y_train, label_map = collector.create_dataset_from_files("train")
    
    # 데이터가 없는 경우, 사용자에게 안내 후 종료
    if len(X_train) == 0:
        print("\n학습 데이터가 없습니다!")
        print("먼저 데이터 수집 모드로 데이터를 수집해주세요:")
        print("  python main.py --mode collect")
        return
    
    # 3. 테스트 데이터 로드 (있을 경우)
    X_test, y_test, _ = collector.create_dataset_from_files("test", label_map=label_map)
    
    # 테스트 데이터가 없는 경우, 학습 데이터 중 일부를 테스트용으로 분리 (8:2 비율)
    if len(X_test) == 0:
        print("테스트 데이터가 없어 학습 데이터를 8:2로 분할합니다.")
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
    
    # 데이터셋 정보 출력
    print(f"\n데이터셋 정보:")
    print(f"  학습 데이터: {X_train.shape}")
    print(f"  테스트 데이터: {X_test.shape}")
    print(f"  클래스 수: {len(label_map)}")
    print(f"  레이블 맵: {label_map}")
    
    # 4. 모델 초기화
    # 입력 시퀀스 (30프레임, 각 프레임당 128차원 특징) 형태로 모델 구성
    feature_dim = collector.hand_detector.FEATURE_DIMENSION * 2
    model = SignLanguageModel(
        input_shape=(30, feature_dim),
        num_classes=len(label_map)
    )
    model.sign_labels = {index: label for label, index in label_map.items()}
    
    # 5. 데이터 증강 (Data Augmentation)
    # 랜덤 노이즈 추가 및 프레임 샘플링을 통해 학습 데이터를 다양화함
    print("\n데이터 증강 중...")
    X_train_aug, y_train_aug = model.create_augmented_data(X_train, y_train)
    print(f"증강 후 학습 데이터: {X_train_aug.shape}")
    
    # 6. 모델 학습 시작
    print("\n모델 학습 시작...")
    history = model.train(
        X_train_aug, y_train_aug,  # 학습 데이터
        X_test, y_test,            # 검증 데이터
        epochs=50,                 # 학습 반복 횟수
        batch_size=32              # 배치 크기
    )
    
    # 7. 모델 평가
    print("\n모델 평가 중...")
    test_loss, test_accuracy, test_top_k = model.model.evaluate(X_test, y_test)
    
    # 테스트 결과 출력
    print(f"\n최종 성능:")
    print(f"  테스트 손실: {test_loss:.4f}")
    print(f"  테스트 정확도: {test_accuracy:.4f}")
    print(f"  Top-3 정확도: {test_top_k:.4f}")
    
    # 8. 모델 및 레이블 저장
    model_path = "models/sign_language_model.h5"
    model.save_model(model_path)
    print(f"\n모델 저장 완료: {model_path}")
    
    # 9. 학습 기록 저장 (loss/accuracy 로그)
    history_path = "models/training_history.json"
    with open(history_path, 'w', encoding='utf-8') as f:
        # numpy 타입은 JSON 직렬화 불가 → float 변환
        history_dict = {
            'loss': [float(x) for x in history.history['loss']],
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_loss': [float(x) for x in history.history.get('val_loss', [])],
            'val_accuracy': [float(x) for x in history.history.get('val_accuracy', [])],
            'label_map': label_map  # 레이블 맵도 저장
        }
        json.dump(history_dict, f, indent=2, ensure_ascii=False)
    
    print(f"학습 기록 저장 완료: {history_path}")
    
    # 10. 학습 결과 시각화
    try:
        import matplotlib.pyplot as plt
        
        # 그래프 크기 설정
        plt.figure(figsize=(12, 4))
        
        # (1) 손실 그래프
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # (2) 정확도 그래프
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        if 'val_accuracy' in history.history:
            plt.plot(history.history['val_accuracy'], label='Val Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # 그래프를 하나의 이미지로 저장
        plt.tight_layout()
        plt.savefig('models/training_plot.png')
        print("학습 그래프 저장 완료: models/training_plot.png")
        
    except ImportError:
        # matplotlib가 설치되지 않은 경우
        print("matplotlib가 설치되지 않아 그래프를 생성할 수 없습니다.")
    
    # 학습 완료 메시지
    print("\n모델 학습이 완료되었습니다!")
    print("이제 다음 명령으로 수어 번역을 시작할 수 있습니다:")
    print("  python main.py --model models/sign_language_model.h5")

if __name__ == "__main__":
    train_sign_language_model()

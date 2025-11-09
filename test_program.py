#!/usr/bin/env python3.11
"""프로그램 실행 테스트"""

import sys
print("🔍 Import 테스트 시작...")

try:
    import cv2
    print("✅ OpenCV import 성공")
except Exception as e:
    print(f"❌ OpenCV import 실패: {e}")
    sys.exit(1)

try:
    import mediapipe as mp
    print("✅ MediaPipe import 성공")
except Exception as e:
    print(f"❌ MediaPipe import 실패: {e}")
    sys.exit(1)

try:
    import tensorflow as tf
    print(f"✅ TensorFlow import 성공 (버전: {tf.__version__})")
except Exception as e:
    print(f"❌ TensorFlow import 실패: {e}")
    sys.exit(1)

try:
    from src.sign_translator import SignLanguageTranslator
    print("✅ SignLanguageTranslator import 성공")
except Exception as e:
    print(f"❌ SignLanguageTranslator import 실패: {e}")
    sys.exit(1)

print("\n🚀 번역기 초기화 테스트...")
try:
    translator = SignLanguageTranslator(model_path="models/sign_language_model.h5")
    print("✅ 번역기 초기화 성공")
    print(f"   - 모델 경로: models/sign_language_model.h5")
    print(f"   - 손 검출기: OK")
    print(f"   - TTS 엔진: {'OK' if translator.tts_available else '비활성화'}")
except Exception as e:
    print(f"❌ 번역기 초기화 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n📷 카메라 접근 테스트...")
try:
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print("✅ 카메라 접근 성공")
        ret, frame = cap.read()
        if ret:
            print(f"   - 프레임 크기: {frame.shape}")
        else:
            print("⚠️  프레임 읽기 실패")
        cap.release()
    else:
        print("❌ 카메라를 열 수 없습니다")
except Exception as e:
    print(f"❌ 카메라 접근 실패: {e}")

print("\n✨ 모든 테스트 완료!")
print("\n사용 방법:")
print("1. python3.11 main.py 실행")
print("2. 카메라 화면이 열리면 **Space 키** 누르기 (번역 시작)")
print("3. '산' 수화 동작을 **0.5초 이상** 유지하기")
print("4. 화면 상단에 '산' 텍스트가 나타나는지 확인")
print("\n주의사항:")
print("- 반드시 Space 키를 눌러야 번역이 시작됩니다!")
print("- 손 동작을 최소 0.5초 이상 유지해야 합니다")
print("- Q 키를 누르면 종료됩니다")
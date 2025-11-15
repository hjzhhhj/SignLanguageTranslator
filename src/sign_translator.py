import cv2
import numpy as np
import pyttsx3
from threading import Thread, Lock
import time
from typing import List, Optional, Tuple
from src.hand_detector import HandDetector
from src.sign_language_model import SignLanguageModel
from PIL import Image, ImageDraw, ImageFont


class SignLanguageTranslator:
    def __init__(self, model_path: Optional[str] = None):
        """
        실시간 수어 번역기 클래스
        Args:
            model_path: 학습된 모델 경로 (없으면 새 모델 생성)
        """
        # 손 인식기 및 수어 인식 모델 초기화
        self.hand_detector = HandDetector()
        self.sign_model = SignLanguageModel(model_path=model_path)

        # 음성 합성(TTS) 엔진 초기화 (실패해도 계속 실행)
        try:
            self.tts_engine = pyttsx3.init()
            self._setup_tts()
            self.tts_available = True
            print("TTS 엔진 초기화 성공")
        except Exception as e:
            self.tts_engine = None
            self.tts_available = False
            print(f"TTS 엔진 초기화 실패 (음성 출력 비활성화): {e}")

        # 번역 관리용 상태 변수
        self.last_prediction = None
        self.last_prediction_time = 0
        self.prediction_cooldown = 2.0  # 같은 단어 반복 방지용 쿨다운(초 단위)

        # 상태 관리 변수
        self.is_running = False
        self.detection_threshold = 0.95  # 예측 신뢰도 임계값 (95% 미만 무시)
        self.min_sequence_length = 15    # 최소 시퀀스 길이

        # UI 관련 상태 변수
        self.display_text = ""
        self.confidence_score = 0.0
        self.text_lock = Lock()  # 스레드 안전성 보장용 락
        self.tts_lock = Lock()   # TTS 엔진 동시 사용 방지용 락

    # 한글 텍스트 렌더링 (PIL 사용)
    def _put_korean_text(self, frame: np.ndarray, text: str, position: tuple,
                         font_size: int = 50, color: tuple = (0, 255, 0)):
        """
        OpenCV 이미지에 한글 텍스트를 렌더링
        Args:
            frame: OpenCV 이미지 (BGR)
            text: 표시할 한글 텍스트
            position: (x, y) 좌표
            font_size: 폰트 크기
            color: 텍스트 색상 (B, G, R)
        Returns:
            텍스트가 그려진 이미지
        """
        # OpenCV BGR -> PIL RGB 변환
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        # 시스템 한글 폰트 사용 (Mac 기준)
        try:
            # Mac 시스템 폰트 경로들
            font_paths = [
                "/System/Library/Fonts/AppleSDGothicNeo.ttc",  # Mac 기본 한글 폰트
                "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
                "/Library/Fonts/AppleGothic.ttf"
            ]

            font = None
            for font_path in font_paths:
                try:
                    font = ImageFont.truetype(font_path, font_size)
                    break
                except:
                    continue

            if font is None:
                # 폰트를 찾지 못하면 기본 폰트 사용
                font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()

        # PIL RGB 색상으로 변환 (BGR -> RGB)
        color_rgb = (color[2], color[1], color[0])
        draw.text(position, text, font=font, fill=color_rgb)

        # PIL RGB -> OpenCV BGR 변환
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # TTS(음성 출력) 설정
    def _setup_tts(self):
        """TTS 엔진 설정 (한국어 음성 지원 시 적용)"""
        voices = self.tts_engine.getProperty('voices')
        for voice in voices:
            # 시스템에 설치된 음성 중 한국어 음성이 있으면 사용
            if 'korean' in voice.name.lower() or 'ko' in voice.id.lower():
                self.tts_engine.setProperty('voice', voice.id)
                break

        # 속도 및 볼륨 설정
        self.tts_engine.setProperty('rate', 150)
        self.tts_engine.setProperty('volume', 0.9)

    def speak_text(self, text: str):
        """텍스트를 음성으로 출력 (비동기 스레드 방식)"""
        if not self.tts_available:
            return  # TTS 사용 불가능하면 무시

        def _speak():
            with self.tts_lock:  # TTS 엔진 동시 사용 방지
                try:
                    self.tts_engine.say(text)
                    self.tts_engine.runAndWait()
                except Exception as e:
                    print(f"TTS 오류: {e}")

        thread = Thread(target=_speak)
        thread.daemon = True
        thread.start()

    # 손 랜드마크 - 특징 벡터 변환
    def _get_padded_feature_vector(self, landmarks_list: List[List[float]]) -> np.ndarray:
        """
        감지된 랜드마크 리스트를 2손(128차원) 기준으로 패딩하여 반환
        (data_collector.py의 동일 로직 복사)
        """
        all_hand_features = []
        feature_dim = self.hand_detector.FEATURE_DIMENSION  # 단일 손의 특징 차원 (예: 64)

        # 감지된 손 개수만큼 특징 추출
        for landmarks in landmarks_list:
            landmarks_array = np.array(landmarks, dtype=np.float32).reshape(-1, 3)
            wrist_x = landmarks_array[0, 0]  # 손목 x좌표 기준으로 좌/우 손 구분
            normalized_features = self.hand_detector.normalize_landmarks(landmarks)
            all_hand_features.append((wrist_x, normalized_features))

        # 손이 한 개만 감지된 경우, 0벡터로 보완
        all_hand_features.sort(key=lambda item: item[0])
        sorted_features = [feat for _, feat in all_hand_features]

        if len(sorted_features) < 2:
            num_missing_hands = 2 - len(sorted_features)
            padding_feature = np.zeros(feature_dim, dtype=np.float32)
            for _ in range(num_missing_hands):
                sorted_features.append(padding_feature)

        # 두 손의 특징을 합쳐 128차원 벡터로 구성
        feature_vector = np.concatenate(sorted_features[:2]).astype(np.float32)
        return feature_vector

    # 단일 프레임 처리
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
        """
        단일 프레임에서 손을 인식하고 수어를 예측
        Returns:
            annotated_frame: 손 인식 표시된 프레임
            translation: 번역 결과 문자열 (없으면 None)
        """
        annotated_frame, landmarks_list = self.hand_detector.detect_hands(frame)
        translation = None

        if landmarks_list:
            # 감지된 손들의 랜드마크를 2손 기준 벡터로 변환
            feature_vector = self._get_padded_feature_vector(landmarks_list)
            self.sign_model.update_sequence_buffer(feature_vector)

            # 일정 길이 이상 시퀀스가 쌓이면 예측 수행
            if len(self.sign_model.sequence_buffer) >= self.min_sequence_length:
                sequence = self.sign_model.get_sequence_array()
                predicted_sign, confidence = self.sign_model.predict_sign(
                    sequence, threshold=self.detection_threshold
                )

                current_time = time.time()

                # 클래스가 1개뿐인 경우 경고: 신뢰도가 높아도 예측하지 않음
                if self.sign_model.num_classes <= 1:
                    # 경고 메시지 표시 (처음 한 번만)
                    if not hasattr(self, '_warned_single_class'):
                        print("\n경고: 학습된 클래스가 1개뿐입니다.")
                        print("더 많은 수어 단어를 수집하고 재학습해주세요:")
                        print("  python3.11 main.py --mode collect")
                        print("  python3.11 main.py --mode train\n")
                        self._warned_single_class = True
                    predicted_sign = None  # 예측 무시

                if predicted_sign and (
                    predicted_sign != self.last_prediction or
                    current_time - self.last_prediction_time > self.prediction_cooldown
                ):
                    translation = predicted_sign
                    self.last_prediction = predicted_sign
                    self.last_prediction_time = current_time

                    # UI 갱신
                    with self.text_lock:
                        self.display_text = predicted_sign
                        self.confidence_score = confidence

                    # 음성 출력
                    self.speak_text(predicted_sign)

        return annotated_frame, translation

    # UI 표시
    def draw_ui(self, frame: np.ndarray) -> np.ndarray:
        """UI 요소(텍스트, 상태, 신뢰도 등)를 프레임 위에 그리기"""
        height, width = frame.shape[:2]

        # 상단 패널 (번역 결과 표시 영역)
        cv2.rectangle(frame, (0, 0), (width, 80), (50, 50, 50), -1)

        with self.text_lock:
            if self.display_text:
                # 중앙에 번역 텍스트 표시 (한글 지원)
                text_x = width // 2 - 50  # 대략 중앙
                frame = self._put_korean_text(frame, self.display_text, (text_x, 30),
                                               font_size=50, color=(0, 255, 0))

                # 신뢰도 표시 (영문이므로 cv2.putText 사용 가능)
                conf_text = f"Confidence: {self.confidence_score:.2%}"
                cv2.putText(frame, conf_text, (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # 하단 상태바 표시
        info_y = height - 30
        cv2.rectangle(frame, (0, height - 40), (width, height), (30, 30, 30), -1)

        # 상태 표시 (실행/일시정지)
        status_text = "Running" if self.is_running else "Paused"
        status_color = (0, 255, 0) if self.is_running else (0, 100, 255)
        cv2.putText(frame, f"Status: {status_text}", (10, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        # 단축키 안내
        cv2.putText(frame, "Space: Start/Stop | Q: Quit | R: Reset | S: Screenshot",
                    (width - 500, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return frame

    # 실시간 번역 실행
    def run_realtime(self):
        """웹캠을 이용한 실시간 수어 번역"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.is_running = True

        print("\n=== 수어 번역 프로그램 시작 ===")
        print("Controls:")
        print("  Space: 번역 시작/중지")
        print("  Q: 종료")
        print("  R: 버퍼 초기화")
        print("  S: 스크린샷 저장\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # 거울 모드

            # 프레임 처리
            if self.is_running:
                processed_frame, translation = self.process_frame(frame)
                if translation:
                    print(f"번역 결과: {translation} (신뢰도: {self.confidence_score:.2%})")
            else:
                processed_frame = frame

            display_frame = self.draw_ui(processed_frame)
            cv2.imshow('Sign Language Translator', display_frame)

            # 키 입력 이벤트 처리
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                self.is_running = not self.is_running
                print(f"번역 {'시작' if self.is_running else '중지'}")
            elif key == ord('r'):
                # 시퀀스 버퍼 및 UI 초기화
                self.sign_model.sequence_buffer.clear()
                with self.text_lock:
                    self.display_text = ""
                    self.confidence_score = 0.0
                print("버퍼 초기화됨")
            elif key == ord('s'):
                # 스크린샷 저장
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"screenshot_{timestamp}.png"
                cv2.imwrite(filename, display_frame)
                print(f"스크린샷 저장됨: {filename}")

        cap.release()
        cv2.destroyAllWindows()
        self.hand_detector.release()
        print("\n프로그램 종료")

    # 비디오 파일 처리
    def process_video_file(self, video_path: str, output_path: Optional[str] = None):
        """
        비디오 파일을 입력으로 받아 수어 번역 수행 및 출력 저장
        """
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        translations = []
        self.is_running = True

        print(f"비디오 처리 중: {video_path}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame, translation = self.process_frame(frame)
            display_frame = self.draw_ui(processed_frame)

            if translation:
                translations.append({
                    'frame': cap.get(cv2.CAP_PROP_POS_FRAMES),
                    'time': cap.get(cv2.CAP_PROP_POS_MSEC) / 1000,
                    'translation': translation
                })

            if out:
                out.write(display_frame)

            # 진행률 표시
            current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            progress = (current_frame / total_frames) * 100
            print(f"진행률: {progress:.1f}%", end='\r')

        cap.release()
        if out:
            out.release()

        print(f"\n비디오 처리 완료. 총 {len(translations)}개 번역 감지됨")
        return translations

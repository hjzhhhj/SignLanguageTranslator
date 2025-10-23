import argparse  
import sys        
import os    
from src.sign_translator import SignLanguageTranslator
from src.data_collector import SignLanguageDataCollector 

def main():
    """
    프로그램의 진입점 (메인 함수)
    실행 모드에 따라: 수어 번역 / 데이터 수집 / 모델 학습 기능 수행
    """
    
    # 명령행 인자 파서 생성
    parser = argparse.ArgumentParser(description='수어 번역 프로그램')

    # 실행 모드 선택 인자: translate / collect / train 중 하나
    parser.add_argument(
        '--mode', type=str,
        choices=['translate', 'collect', 'train'],  # 허용 모드
        default='translate',                        # 기본 모드
        help='실행 모드 선택 (translate: 번역, collect: 데이터 수집, train: 모델 학습)'
    )

    # 사용할 모델 파일 경로 지정
    parser.add_argument(
        '--model', type=str, default=None,
        help='사용할 모델 파일 경로'
    )

    # 입력 비디오 파일 경로 지정 (없으면 웹캠 사용)
    parser.add_argument(
        '--video', type=str, default=None,
        help='처리할 비디오 파일 경로 (없으면 웹캠 사용)'
    )

    # 처리 결과를 저장할 출력 비디오 경로
    parser.add_argument(
        '--output', type=str, default=None,
        help='출력 비디오 파일 경로'
    )

    # 인자 파싱 실행
    args = parser.parse_args()

    # 1. 번역 모드
    if args.mode == 'translate':
        print("\n" + "="*50)
        print("       수어 번역 프로그램 ")
        print("="*50)

        # 모델 경로 확인 (존재하지 않으면 경고 출력 후 None으로 처리)
        model_path = args.model
        if model_path and not os.path.exists(model_path):
            print(f"경고: 모델 파일을 찾을 수 없습니다: {model_path}")
            print("사전 정의된 수어 단어만 인식 가능합니다.")
            model_path = None

        # SignLanguageTranslator 객체 생성 (수어 번역기 초기화)
        translator = SignLanguageTranslator(model_path=model_path)

        # --- (1) 비디오 파일 입력 시 ---
        if args.video:
            # 비디오 파일 존재 여부 확인
            if not os.path.exists(args.video):
                print(f"오류: 비디오 파일을 찾을 수 없습니다: {args.video}")
                sys.exit(1)  # 프로그램 종료

            # 비디오 파일 처리 실행 (수어 인식 및 결과 수집)
            translations = translator.process_video_file(args.video, args.output)

            # 처리 후 번역 결과 요약 출력
            print("\n번역 결과:")
            print("-" * 40)
            for trans in translations:
                print(f"시간: {trans['time']:.2f}초 - {trans['translation']}")

        # --- (2) 웹캠 실시간 번역 ---
        else:
            try:
                translator.run_realtime()  # 실시간 수어 번역 실행
            except KeyboardInterrupt:
                # 사용자가 Ctrl+C로 종료 시
                print("\n프로그램 종료")
            except Exception as e:
                # 예기치 않은 예외 발생 시
                print(f"오류 발생: {e}")
                sys.exit(1)

    # 2. 데이터 수집 모드
    elif args.mode == 'collect':
        print("\n" + "="*50)
        print("      수어 데이터 수집 모드")
        print("="*50)

        # 데이터 수집기 초기화
        collector = SignLanguageDataCollector()

        try:
            # 대화형 수집 모드 실행 (카메라 통해 수어 데이터 수집)
            collector.interactive_collection()
        except KeyboardInterrupt:
            # 사용자 강제 종료 시
            print("\n데이터 수집 종료")

    # 3. 모델 학습 모드
    elif args.mode == 'train':
        print("\n" + "="*50)
        print("       모델 학습 모드")
        print("="*50)

        # 학습 함수 불러오기
        from train_model import train_sign_language_model

        try:
            # 모델 학습 실행
            train_sign_language_model()
        except ImportError:
            # 학습 모듈이 없을 경우
            print("학습 모듈을 찾을 수 없습니다.")
            print("train_model.py 파일이 필요합니다.")
            sys.exit(1)
        except Exception as e:
            # 학습 중 예외 처리
            print(f"학습 중 오류 발생: {e}")
            sys.exit(1)

    # 모든 모드 종료 후 공통 출력
    print("\n프로그램을 종료합니다.")


if __name__ == "__main__":
    main()

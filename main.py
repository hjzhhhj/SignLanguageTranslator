#!/usr/bin/env python3
"""
수어 번역 프로그램 메인 애플리케이션
"""
import argparse
import sys
import os
from src.sign_translator import SignLanguageTranslator
from src.data_collector import SignLanguageDataCollector

def main():
    parser = argparse.ArgumentParser(description='수어 번역 프로그램')
    parser.add_argument('--mode', type=str, choices=['translate', 'collect', 'train'],
                       default='translate',
                       help='실행 모드 선택 (translate: 번역, collect: 데이터 수집, train: 모델 학습)')
    parser.add_argument('--model', type=str, default=None,
                       help='사용할 모델 파일 경로')
    parser.add_argument('--video', type=str, default=None,
                       help='처리할 비디오 파일 경로 (없으면 웹캠 사용)')
    parser.add_argument('--output', type=str, default=None,
                       help='출력 비디오 파일 경로')

    args = parser.parse_args()

    if args.mode == 'translate':
        # 번역 모드
        print("\n" + "="*50)
        print("       수어 번역 프로그램 ")
        print("="*50)

        # 모델 경로 확인
        model_path = args.model
        if model_path and not os.path.exists(model_path):
            print(f"경고: 모델 파일을 찾을 수 없습니다: {model_path}")
            print("사전 정의된 수어 단어만 인식 가능합니다.")
            model_path = None

        # 번역기 초기화
        translator = SignLanguageTranslator(model_path=model_path)

        if args.video:
            # 비디오 파일 처리
            if not os.path.exists(args.video):
                print(f"오류: 비디오 파일을 찾을 수 없습니다: {args.video}")
                sys.exit(1)

            translations = translator.process_video_file(args.video, args.output)

            # 번역 결과 출력
            print("\n번역 결과:")
            print("-" * 40)
            for trans in translations:
                print(f"시간: {trans['time']:.2f}초 - {trans['translation']}")
        else:
            # 실시간 웹캠 번역
            try:
                translator.run_realtime()
            except KeyboardInterrupt:
                print("\n프로그램 종료")
            except Exception as e:
                print(f"오류 발생: {e}")
                sys.exit(1)

    elif args.mode == 'collect':
        # 데이터 수집 모드
        print("\n" + "="*50)
        print("      수어 데이터 수집 모드")
        print("="*50)

        collector = SignLanguageDataCollector()
        try:
            collector.interactive_collection()
        except KeyboardInterrupt:
            print("\n데이터 수집 종료")

    elif args.mode == 'train':
        # 모델 학습 모드
        print("\n" + "="*50)
        print("       모델 학습 모드")
        print("="*50)

        from train_model import train_sign_language_model

        try:
            train_sign_language_model()
        except ImportError:
            print("학습 모듈을 찾을 수 없습니다.")
            print("train_model.py 파일이 필요합니다.")
            sys.exit(1)
        except Exception as e:
            print(f"학습 중 오류 발생: {e}")
            sys.exit(1)

    print("\n프로그램을 종료합니다.")

if __name__ == "__main__":
    main()

import argparse
import netron


def parse_args():
    parser = argparse.ArgumentParser(description="ONNX visualization")
    parser.add_argument("onnx")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # 127.0.0.1:8080
    netron.start(args.onnx, browse=False)


if __name__ == "__main__":
    main()

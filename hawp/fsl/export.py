import argparse

import numpy as np
import onnx
import onnxruntime as ort
import torch.onnx
from torch.utils.data import DataLoader

from .config import cfg
from .dataset.build import build_transform
from .model.build import build_model
from .predict import ImageList


def parse_args():
    parser = argparse.ArgumentParser(
        prog="python -m hawp.fsl.export",
        usage="%(prog)s [options] config image",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("config", help="the path of config file")
    parser.add_argument("image", help="input image")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument(
        "-o", "--output", default="hawp-fsl.onnx", nargs="?", help="Path at which to write the exported model"
    )

    return parser.parse_args()


def to_numpy(tensor):
    if isinstance(tensor, int):
        return tensor
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def check_model(model_file):
    model = onnx.load(model_file)
    onnx.checker.check_model(model)


def verify_model(model_file, input, output):
    ort_session = ort.InferenceSession(model_file, providers=["CPUExecutionProvider"]) #, providers=["CUDAExecutionProvider"])
    ort_inputs = {"image": to_numpy(input)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    for obs, exp in zip(ort_outs, output):
        try:
            np.testing.assert_allclose(to_numpy(exp), obs, rtol=1e-03, atol=1e-05)
        except Exception as e:
            print(e)
            print(obs)
            print(exp)


def get_model(cfg, checkpoint):
    device = cfg.MODEL.DEVICE
    model = build_model(cfg).to(device)
    # model.use_residual = 2

    ckpt = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.export_mode = True
    return model.eval()


def get_input(cfg, input_file):
    device = cfg.MODEL.DEVICE
    transform = build_transform(cfg)
    dataset = ImageList([input_file], transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, pin_memory=True)
    for tensor, meta in dataloader:
        yield tensor.to(device), meta


def export(cfg, checkpoint, input_file, output_file):
    model = get_model(cfg, checkpoint)
    [input] = get_input(cfg, input_file)
    output = model(*input)
    output_names = [
        "vertices",
        "v_confidences",
        "edges",
        "edge_weights",
        "frame_width",
        "frame_height",
    ]
    torch.onnx.export(
        model, input, output_file, opset_version=14, input_names=["image"], output_names=output_names,
    )

    return input, output

def main():
    args = parse_args()
    cfg.merge_from_file(args.config)
    cfg.freeze()

    input, output = export(cfg, args.ckpt, args.image, args.output)
    check_model(args.output)
    img, _ = input
    verify_model(args.output, img, output)

if __name__ == "__main__":
    main()

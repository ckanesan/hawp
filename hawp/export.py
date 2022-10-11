import argparse

import numpy as np
import onnx
import onnxruntime as ort
import torch.onnx

from .config import cfg
from .predicting import WireframeParser


def cli():
    parser = argparse.ArgumentParser(
        prog="python -m hawp.export",
        usage="%(prog)s [options] image",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("image", help="input image")
    parser.add_argument(
        "-o", "--output", default="hawp.onnx", nargs="?", help="Path at which to write the exported model"
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
    ort_session = ort.InferenceSession(model_file, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    ort_inputs = {"image": to_numpy(input)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    for idx, (obs, exp) in enumerate(zip(ort_outs, output)):
        np.testing.assert_allclose(to_numpy(exp), obs, rtol=1e-03, atol=1e-05, err_msg=f"Failed tensor {idx}")


def export(input_file, output_file):
    wireframe_parser = WireframeParser(export_mode=True)
    for _ in wireframe_parser.images([input_file]):
        pass

    model = wireframe_parser.model
    [input] = wireframe_parser.inputs
    [output] = wireframe_parser.outputs
    output_names = [
        "vertices",
        "v_confidences",
        "edges",
        "edge_weights",
        "frame_width",
        "frame_height",
    ]
    torch.onnx.export(model, input, output_file, opset_version=11, input_names=["image"], output_names=output_names)

    return input, output


if __name__ == "__main__":
    cfg.freeze()
    args = cli()

    input, output = export(args.image, args.output)
    check_model(args.output)
    img, _ = input
    verify_model(args.output, img, output)

import numpy as np
import onnx
import onnxruntime as ort
import torch.onnx

from .config import cfg
from .predicting import WireframeParser


INPUT_FILE = "/home/ckanesan/Data/semantic-keypoints/826840/Image_000001.jpg"
OUTPUT_FILE = "hawp.onnx"

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def check_model():
    model = onnx.load(OUTPUT_FILE)
    onnx.checker.check_model(model)

    inferred_model = onnx.shape_inference.infer_shapes(model)
    onnx.save(inferred_model, "hawp_inf.onnx")

def verify_model(input, output):
    ort_session = ort.InferenceSession(OUTPUT_FILE, providers=["CUDAExecutionProvider"])
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(output), ort_outs[0], rtol=1e-03, atol=1e-05)


def export(cfg):
    wireframe_parser = WireframeParser(export_mode=True)
    for _ in wireframe_parser.images([INPUT_FILE]):
        pass

    model = wireframe_parser.model
    [input] = wireframe_parser.inputs
    [output] = wireframe_parser.outputs
    torch.onnx.export(model, input, OUTPUT_FILE, opset_version=11, )

    return input, output


if __name__ == "__main__":
    cfg.freeze()
    input, output = export(cfg)
    check_model()
    #verify_model(input, output)


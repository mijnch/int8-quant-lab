"""LiteRT 인터프리터를 세 가지 커널 경로로 실행한다.

- reference : BUILTIN_REF — 참조 커널(단순 루프, 사양서 역할)
- optimized : BUILTIN_WITHOUT_DEFAULT_DELEGATES — 최적화 커널, XNNPACK 끔
- xnnpack   : BUILTIN — 기본 설정. XNNPACK 위임(delegate)이 지원 연산을 가져간다
"""
import numpy as np
from ai_edge_litert.interpreter import Interpreter, OpResolverType

MODES = {
    "reference": OpResolverType.BUILTIN_REF,
    "optimized": OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
    "xnnpack": OpResolverType.BUILTIN,
}


def make(model, mode, *, batch=None, preserve=False, split_delegate=False, num_threads=1):
    """model: 경로 또는 bytes. split_delegate=True면 XNNPACK이 연산마다 따로 위임되어
    (노드 융합 끔) 각 연산의 출력이 그래프 텐서로 남는다 — 층별 대조용."""
    kw = dict(experimental_op_resolver_type=MODES[mode], num_threads=num_threads,
              experimental_preserve_all_tensors=preserve)
    if split_delegate:
        kw["experimental_disable_delegate_node_fusion"] = True
    if isinstance(model, (bytes, bytearray)):
        it = Interpreter(model_content=bytes(model), **kw)
    else:
        it = Interpreter(model_path=str(model), **kw)
    if batch is not None:
        d = it.get_input_details()[0]
        it.resize_tensor_input(d["index"], [batch] + list(d["shape"][1:]), strict=False)
    it.allocate_tensors()
    return it


def invoke(it, x):
    it.set_tensor(it.get_input_details()[0]["index"], x)
    it.invoke()
    return it.get_tensor(it.get_output_details()[0]["index"])


def run(model, mode, x, batch=512, **kw):
    """x 전체를 batch 단위로 나눠 실행해 출력을 이어 붙인다."""
    outs = []
    it = None
    for s in range(0, len(x), batch):
        chunk = x[s:s + batch]
        if it is None or len(chunk) != it.get_input_details()[0]["shape"][0]:
            it = make(model, mode, batch=len(chunk), **kw)
        outs.append(invoke(it, chunk).copy())
    return np.concatenate(outs)


def execution_plan(it):
    """실행된 노드 이름 목록(위임 노드 포함)."""
    return [d["op_name"] for d in it._get_ops_details()]

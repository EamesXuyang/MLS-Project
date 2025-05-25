from ..tensor import Tensor
import numpy as np
import base64
import io
from typing import Dict


def decode_parameters(params: Dict[str, Tensor]) -> None:
    decode_params = {}

    for key, value in params.items():
        decode_params[key] = Tensor(np.load(io.BytesIO(base64.b64decode(value))))

    return decode_params

def encode_parameters(params: Dict[str, Tensor]) -> Dict[str, str]:
    encode_params = {}

    for key, value in params.items():
        value = value.data
        buffer = io.BytesIO()
        np.save(buffer, value)
        encode_params[key] = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return encode_params
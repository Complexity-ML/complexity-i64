"""
Complexity-I64: Integer-native transformer architecture.

Train in float32. Deploy in INT8.

INL - 2025
"""

from complexity_i64.models.config import I64Config
from complexity_i64.models.modeling import I64Model, create_i64_model

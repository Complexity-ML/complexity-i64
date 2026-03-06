"""
Complexity-I64: Integer-native transformer architecture.

Every matmul is INT8. Float only where mathematically irreducible
(RoPE rotation, attention softmax, RMSNorm rsqrt).

INL - 2025
"""

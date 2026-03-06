from setuptools import setup, find_packages

setup(
    name="complexity-i64",
    version="0.1.0",
    description="Integer-native Complexity architecture — Train float32, deploy INT8",
    author="INL",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.2.0",
        "transformers>=4.30.0",
        "datasets>=2.0.0",
        "tokenizers>=0.13.0",
        "jinja2>=3.0",
        "tqdm",
        "tensorboard",
        "pyyaml",
    ],
    extras_require={
        "lora": ["peft>=0.7.0"],
        "triton": ["triton>=2.0.0"],
        "tokenizer": ["complexity-framework"],
    },
)

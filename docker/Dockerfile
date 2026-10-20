# Optional container: official vLLM 0.29.0 image + the extra packages of this repository.
#   docker build -t selectrl-qwen3-8b -f docker/Dockerfile .
#   docker run --gpus all --ipc=host --network host -v $PWD:/workspace -w /workspace -it selectrl-qwen3-8b bash
FROM vllm/vllm-openai:v0.29.0
WORKDIR /workspace
COPY requirements/extra.txt /tmp/extra.txt
RUN python3 -m pip freeze | grep -i -E '^(torch|vllm|triton|transformers|numpy)==' > /tmp/cons.txt && \
    python3 -m pip install --no-cache-dir -c /tmp/cons.txt -r /tmp/extra.txt
ENTRYPOINT []
CMD ["bash"]

FROM public.ecr.aws/x8v8d7g8/mars-base:latest
WORKDIR /app

# Copy repository contents
COPY . .

# Install project and dependencies using tools present in the base image
# Avoid upgrading or installing package managers; ensure fully offline-ready env
# Install into the system interpreter inside the container (no venv required)
RUN uv pip install --system -e . \
    && uv pip install --system -r requirements/typing.txt \
    && uv pip install --system -r requirements/pytorch/base.txt \
    && uv pip install --system -r requirements/fabric/base.txt \
    && uv pip install --system pytest

# Provide an interactive shell as the default entrypoint
CMD ["/bin/bash"]

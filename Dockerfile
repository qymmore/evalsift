# Small runtime image: `docker run --rm -v "$PWD/logs:/logs" evalsift check /logs`
FROM python:3.12-slim AS build
WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip wheel --no-cache-dir --wheel-dir /wheels .

FROM python:3.12-slim
RUN useradd --create-home --uid 1000 sift
COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels
USER sift
WORKDIR /work
ENTRYPOINT ["evalsift"]
CMD ["--help"]

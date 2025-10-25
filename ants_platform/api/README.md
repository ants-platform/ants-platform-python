# Ants Platform Python Library

[![fern shield](https://img.shields.io/badge/%F0%9F%8C%BF-Built%20with%20Fern-brightgreen)](https://buildwithfern.com?utm_source=github&utm_medium=github&utm_campaign=readme&utm_source=ants-platform%2FPython)
[![pypi](https://img.shields.io/pypi/v/ants-platform)](https://pypi.python.org/pypi/ants-platform)

The Ants Platform Python library provides convenient access to the Ants Platform API from Python.

## Installation

```sh
pip install ants-platform
```

## Usage

Instantiate and use the client with the following:

```python
from ants_platform import CreateAnnotationQueueRequest
from ants_platform.client import FernAntsPlatform

client = FernAntsPlatform(
    x_ants_platform_sdk_name="YOUR_X_ANTS_PLATFORM_SDK_NAME",
    x_ants_platform_sdk_version="YOUR_X_ANTS_PLATFORM_SDK_VERSION",
    x_ants_platform_public_key="YOUR_X_ANTS_PLATFORM_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    base_url="https://yourhost.com/path/to/api",
)
client.annotation_queues.create_queue(
    request=CreateAnnotationQueueRequest(
        name="name",
        score_config_ids=["scoreConfigIds", "scoreConfigIds"],
    ),
)
```

## Async Client

The SDK also exports an `async` client so that you can make non-blocking calls to our API.

```python
import asyncio

from ants_platform import CreateAnnotationQueueRequest
from ants_platform.client import AsyncFernAntsPlatform

client = AsyncFernAntsPlatform(
    x_ants_platform_sdk_name="YOUR_X_ANTS_PLATFORM_SDK_NAME",
    x_ants_platform_sdk_version="YOUR_X_ANTS_PLATFORM_SDK_VERSION",
    x_ants_platform_public_key="YOUR_X_ANTS_PLATFORM_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    base_url="https://yourhost.com/path/to/api",
)


async def main() -> None:
    await client.annotation_queues.create_queue(
        request=CreateAnnotationQueueRequest(
            name="name",
            score_config_ids=["scoreConfigIds", "scoreConfigIds"],
        ),
    )


asyncio.run(main())
```

## Exception Handling

When the API returns a non-success status code (4xx or 5xx response), a subclass of the following error
will be thrown.

```python
from .api_error import ApiError

try:
    client.annotation_queues.create_queue(...)
except ApiError as e:
    print(e.status_code)
    print(e.body)
```

## Advanced

### Retries

The SDK is instrumented with automatic retries with exponential backoff. A request will be retried as long
as the request is deemed retriable and the number of retry attempts has not grown larger than the configured
retry limit (default: 2).

A request is deemed retriable when any of the following HTTP status codes is returned:

- [408](https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/408) (Timeout)
- [429](https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/429) (Too Many Requests)
- [5XX](https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/500) (Internal Server Errors)

Use the `max_retries` request option to configure this behavior.

```python
client.annotation_queues.create_queue(...,{
    max_retries=1
})
```

### Timeouts

The SDK defaults to a 60 second timeout. You can configure this with a timeout option at the client or request level.

```python

from ants_platform.client import FernAntsPlatform

client = FernAntsPlatform(..., { timeout=20.0 }, )


# Override timeout for a specific method
client.annotation_queues.create_queue(...,{
    timeout_in_seconds=1
})
```

### Custom Client

You can override the `httpx` client to customize it for your use-case. Some common use-cases include support for proxies
and transports.

```python
import httpx
from ants_platform.client import FernAntsPlatform

client = FernAntsPlatform(
    ...,
    http_client=httpx.Client(
        proxies="http://my.test.proxy.example.com",
        transport=httpx.HTTPTransport(local_address="0.0.0.0"),
    ),
)
```

## Contributing

While we value open-source contributions to this SDK, this library is generated programmatically.
Additions made directly to this library would have to be moved over to our generation code,
otherwise they would be overwritten upon the next generated release. Feel free to open a PR as
a proof of concept, but know that we will not be able to merge it as-is. We suggest opening
an issue first to discuss with us!

On the other hand, contributions to the README are always very welcome!

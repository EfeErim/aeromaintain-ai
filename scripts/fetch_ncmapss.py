"""Fetch selected members of the NASA archive using HTTP ranges, not all 16 GB."""

import argparse
import io
import json
import shutil
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

URL = "https://phm-datasets.s3.amazonaws.com/NASA/17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2.zip"


class RemoteArchive(io.RawIOBase):
    def __init__(self):
        response = urllib.request.urlopen(
            urllib.request.Request(URL, method="HEAD"), timeout=60
        )
        self.size = int(response.headers["Content-Length"])
        self.etag = response.headers["ETag"]
        self.position = 0

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self.position = offset + (
            self.position if whence == 1 else self.size if whence == 2 else 0
        )
        return self.position

    def tell(self):
        return self.position

    def read(self, size=-1):
        size = (
            self.size - self.position
            if size < 0
            else min(size, self.size - self.position)
        )
        if size == 0:
            return b""
        request = urllib.request.Request(
            URL,
            headers={
                "Range": f"bytes={self.position}-{self.position + size - 1}",
                "If-Match": self.etag,
            },
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            if response.status != 206:
                raise RuntimeError("Server did not honor byte range")
            payload = response.read()
        if len(payload) != size:
            raise RuntimeError("Incomplete byte range")
        self.position += size
        return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/raw/ncmapss"))
    parser.add_argument("--subsets", nargs="+", default=["DS02", "DS08"])
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    remote = RemoteArchive()
    args.output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(remote) as listing:
        nested = [i for i in listing.infolist() if i.filename.endswith(".zip")]
    if nested and not args.list:
        cached = args.output / "nasa_archive.zip"
        partial_archive = cached.with_suffix(".zip.partial")
        if not cached.exists():
            block = 16 * 1024 * 1024
            with partial_archive.open("wb") as handle:
                handle.truncate(remote.size)

            def fetch(offset):
                end = min(remote.size, offset + block) - 1
                request = urllib.request.Request(
                    URL,
                    headers={"Range": f"bytes={offset}-{end}", "If-Match": remote.etag},
                )
                with urllib.request.urlopen(request, timeout=180) as response:
                    if response.status != 206:
                        raise RuntimeError("Server ignored range")
                    payload = response.read()
                if len(payload) != end - offset + 1:
                    raise RuntimeError("Truncated archive chunk")
                with partial_archive.open("r+b") as handle:
                    handle.seek(offset)
                    handle.write(payload)

            with ThreadPoolExecutor(max_workers=8) as pool:
                for number, _ in enumerate(
                    pool.map(fetch, range(0, remote.size, block))
                ):
                    if number % 32 == 0:
                        progress = min((number + 1) * block, remote.size) / remote.size
                        print(
                            f"Archive {progress:.1%}",
                            flush=True,
                        )
            partial_archive.replace(cached)
        inner = args.output / "data_set.zip"
        if not inner.exists():
            with (
                zipfile.ZipFile(cached) as outer,
                outer.open(nested[0]) as source,
                inner.with_suffix(".partial").open("wb") as target,
            ):
                shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
            inner.with_suffix(".partial").replace(inner)
        archive_source = inner
    else:
        archive_source = remote
    with zipfile.ZipFile(archive_source) as archive:
        members = [item for item in archive.infolist() if item.filename.endswith(".h5")]
        print(
            json.dumps(
                [
                    {
                        "name": item.filename,
                        "compressed": item.compress_size,
                        "bytes": item.file_size,
                    }
                    for item in members
                ]
            ),
            flush=True,
        )
        if args.list:
            return
        args.output.mkdir(parents=True, exist_ok=True)
        for item in members:
            if not any(subset in item.filename for subset in args.subsets):
                continue
            target = args.output / Path(item.filename).name
            if target.exists():
                raise FileExistsError(target)
            partial = target.with_suffix(".h5.partial")
            print(f"Downloading {item.filename}", flush=True)
            with archive.open(item) as source, partial.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
            partial.replace(target)
            print(f"Completed {target}", flush=True)
        (args.output / "source.json").write_text(
            json.dumps(
                {"url": URL, "etag": remote.etag, "archive_bytes": remote.size},
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()

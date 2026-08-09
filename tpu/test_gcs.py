"""Smoke-test read/write access to the diffjax GCS bucket.

Run with Application Default Credentials available, for example:

    python tpu/test_gcs.py
"""

import argparse
import uuid

from google.cloud import storage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default="diffjax", help="GCS bucket name.")
    parser.add_argument(
        "--object",
        default=None,
        help="Object name under the bucket. Defaults to a unique tpu/ path.",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep the test object instead of deleting it after verification.",
    )
    args = parser.parse_args()

    object_name = args.object or f"tpu/gcs_read_write_test_{uuid.uuid4().hex}.txt"
    payload = f"GCS read/write test object: {object_name}\n".encode("utf-8")
    blob = storage.Client().bucket(args.bucket).blob(object_name)
    uploaded = False

    try:
        print(f"Writing gs://{args.bucket}/{object_name} ...")
        blob.upload_from_string(payload, content_type="text/plain")
        uploaded = True

        print(f"Reading gs://{args.bucket}/{object_name} ...")
        received = blob.download_as_bytes()
        if received != payload:
            raise RuntimeError("GCS read/write verification failed: content mismatch")

        print("GCS read/write test passed.")
    finally:
        if uploaded and not args.keep:
            blob.delete()
            print(f"Deleted test object gs://{args.bucket}/{object_name}.")


if __name__ == "__main__":
    main()

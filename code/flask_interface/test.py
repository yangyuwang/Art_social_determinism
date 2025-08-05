import os
import boto3
from botocore.client import Config

def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["MINIO_URL"],  # e.g., https://mpibberlin.com
        aws_access_key_id=os.environ["MINIO_ACCESS_KEY"],
        aws_secret_access_key=os.environ["MINIO_SECRET_KEY"],
        config=Config(signature_version="s3v4")
    )

def list_objects_in_bucket():
    bucket = os.environ.get("MINIO_BUCKET")
    if not bucket:
        raise ValueError("MINIO_BUCKET must be set in the environment.")

    s3 = get_s3_client()

    print(f"📦 Checking bucket: {bucket}")
    try:
        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket)

        found = False
        for page in pages:
            contents = page.get("Contents", [])
            if contents:
                found = True
                for obj in contents:
                    print("🖼️", obj["Key"])

        if not found:
            print("📭 Bucket is empty.")

    except Exception as e:
        print("❌ Error accessing bucket:", e)

if __name__ == "__main__":
    list_objects_in_bucket()

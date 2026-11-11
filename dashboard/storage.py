"""Google Cloud Storage helpers."""

from pathlib import Path

from google.cloud import storage


def upload_dir(local_dir: Path, bucket: str, prefix: str = '') -> list[str]:
    """Upload every file directly inside `local_dir` to a GCS bucket.

    Objects are overwritten if they already exist. Subdirectories are ignored.

    Args:
        local_dir: directory whose files are uploaded
        bucket: destination bucket name
        prefix: optional object-name prefix, e.g. 'dashboard-data'

    Returns: gs:// URIs of the uploaded objects
    """
    client = storage.Client()
    bucket_obj = client.bucket(bucket)
    prefix = prefix.strip('/')

    uris = []
    for path in sorted(local_dir.iterdir()):
        if not path.is_file():
            continue
        key = f'{prefix}/{path.name}' if prefix else path.name
        bucket_obj.blob(key).upload_from_filename(path)
        uris.append(f'gs://{bucket}/{key}')
    return uris

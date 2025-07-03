from PIL import Image
import os
import tempfile


def image_to_pdf_main(minio_client, catalog_path, origin_file_name):
    none_ext_file_name = os.path.splitext(origin_file_name)[0]
    split_path = catalog_path.split('/', 1)
    bucket_name = split_path[0]
    dir_path = split_path[1] if len(split_path) > 1 else ""
    pdf_object_name = f"{dir_path}/{none_ext_file_name}.pdf"

    pdf_file_path = image_to_pdf(minio_client, bucket_name, dir_path, origin_file_name)

    minio_client.fput_object(bucket_name, pdf_object_name, pdf_file_path)
    presigned_url = minio_client.presigned_get_object(bucket_name, pdf_object_name)

    return presigned_url

def image_to_pdf(minio_client, bucket_name, dir_path, origin_file_name):
    ext = os.path.splitext(origin_file_name)[1]
    temp_dir = tempfile.mkdtemp()
    base_name = os.path.splitext(os.path.basename(f"{dir_path}/{origin_file_name}"))[0]
    temp_file_path = os.path.join(temp_dir, f"{base_name}.{ext}")
    minio_client.fget_object(bucket_name, f"{dir_path}/{origin_file_name}", temp_file_path)

    img = Image.open(temp_file_path)
    pdf_file_path = os.path.join(temp_dir, f"{os.path.splitext(os.path.basename(temp_file_path))[0]}.pdf")

    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        img = img.convert('RGB')

    img.save(pdf_file_path, "PDF", resolution=400.0)
    return pdf_file_path

import logging
from logging.handlers import RotatingFileHandler

# 로거 설정 함수
def setup_logger(name, log_file, level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.propagate = False

    return logger

# 각 라우터에 대한 로거 설정
models_embedding_logger = setup_logger('models_embedding', '/home/pjtl2w01admin/btw/fast-api/log/models/embedding/models_embedding.log')
models_segmentation_logger = setup_logger('models_segmentation', '/home/pjtl2w01admin/btw/fast-api/log/models/segmentation/models_segmentation.log')
models_summarize_logger = setup_logger('models_summarize', '/home/pjtl2w01admin/btw/fast-api/log/models/summarize/models_summarize.log')

extraction_hwp_logger = setup_logger('extraction_hwp', '/home/pjtl2w01admin/btw/fast-api/log/extraction/hwp_to_text/extraction_hwp.log')
extraction_image_logger = setup_logger('extraction_image', '/home/pjtl2w01admin/btw/fast-api/log/extraction/image_ocr/extraction_image.log')
extraction_pdf_logger = setup_logger('extraction_pdf', '/home/pjtl2w01admin/btw/fast-api/log/extraction/pdf_to_text/extraction_pdf.log')

converter_docx_logger = setup_logger('converter_docx', '/home/pjtl2w01admin/btw/fast-api/log/converter/docx_to_pdf/converter_docx.log')
converter_excel_to_csv_logger = setup_logger('converter_excel_to_csv', '/home/pjtl2w01admin/btw/fast-api/log/converter/excel_to_csv/converter_excel_to_csv.log') # logging 설정 x
converter_excel_to_pdf_logger = setup_logger('converter_excel_to_pdf', '/home/pjtl2w01admin/btw/fast-api/log/converter/excel_to_pdf/converter_excel_to_pdf.log')
converter_image_logger = setup_logger('converter_image', '/home/pjtl2w01admin/btw/fast-api/log/converter/image_to_pdf/converter_image.log')
converter_ppt_logger = setup_logger('converter_ppt', '/home/pjtl2w01admin/btw/fast-api/log/converter/ppt_to_pdf/converter_ppt.log')
converter_hwp_logger = setup_logger('converter_hwp', '/home/pjtl2w01admin/btw/fast-api/log/converter/hwp_to_xhtml/converter_hwp.log') # logging 설정 x
converter_xhtml_logger = setup_logger('converter_xhtml', '/home/pjtl2w01admin/btw/fast-api/log/converter/xhtml_to_pdf/converter_xhtml.log')
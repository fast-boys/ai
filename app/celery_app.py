# celery_app.py
from celery import Celery, shared_task
from celery.signals import worker_process_init
import base64
import os
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
import torch
from sentence_transformers import SentenceTransformer
       
CELERY_REDIS = "redis://35.206.237.189:5672"

# celery_core = Celery(
#     'core_worker',
#     broker=f'{CELERY_REDIS}/0',
#     backend=f'{CELERY_REDIS}/1'
# )

celery_ai = Celery(
    'ai_worker',
    broker=f'{CELERY_REDIS}/0',
    backend=f'{CELERY_REDIS}/1',
    include=["app.celery_app"]
)
celery_ai.autodiscover_tasks()
celery_ai.conf.task_default_queue = "ai_to_core_queue"


@worker_process_init.connect
def env_settings(*args, **kwargs):
    # SSAFY에서 받은 GPU 사용하도록 설정
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"

    
@shared_task(name="ai_worker.vector_embedding", queue="core_to_ai_queue")
def vector_embedding(**kwargs):
    # 순서가 뒤섞이는 경우를 방지하기 위한 kwargs 도입
    url_id = kwargs.get('url_id', None)
    raw_text = kwargs.get('raw_text', None)
    
    if url_id is None or raw_text is None:
        return {"status": "Error", "message": "인자 파싱 실패."}
    
    # 모델 초기화
    nltk.download('punkt')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
    model = model.to(device)
    print('Model loaded. Device:', device)
    
    sentences = sent_tokenize(raw_text)
    sentence_embeddings = model.encode(sentences)
    sentence_embeddings = [torch.tensor(embedding) for embedding in sentence_embeddings]
    document_embedding = torch.mean(torch.stack(sentence_embeddings), dim=0)
    
    serialized_data = {
        'url_id': url_id,
        'vector': document_embedding.tolist(),  # 리스트 데이터
    }
    
    # 결과 데이터를 Core 서버의 worker에게 전달
    task = celery_ai.send_task("core_worker.process_data", args=[serialized_data])
    return {"message": "전달 완료. [ai -> core]", "task_id": task.id}


if __name__ == '__main__':
    celery_ai.worker_main(argv=['worker', '--loglevel=info', '--concurrency=4', '--queues=core_to_ai_queue'])
